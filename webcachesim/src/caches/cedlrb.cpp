//
// Created by cedrick on 12/1/21.
//

#include "cedlrb.h"
#include <algorithm>
#include "utils.h"
#include <chrono>

// sample
// admit
// evict
// forget
// lookup

// rank

using namespace chrono;
using namespace std;
using namespace cedlrb;

void CedLRBCache::train() {
    ++n_retrain;
    auto timeBegin = chrono::system_clock::now();
    if (booster) LGBM_BoosterFree(booster);
    // create training dataset
    DatasetHandle trainData;
    LGBM_DatasetCreateFromCSR(
            static_cast<void *>(training_data->indptr.data()),
            C_API_DTYPE_INT32,
            training_data->indices.data(),
            static_cast<void *>(training_data->data.data()),
            C_API_DTYPE_FLOAT64,
            training_data->indptr.size(),
            training_data->data.size(),
            n_feature,  //remove future t
            training_params,
            nullptr,
            &trainData);

    LGBM_DatasetSetField(trainData,
                         "label",
                         static_cast<void *>(training_data->labels.data()),
                         training_data->labels.size(),
                         C_API_DTYPE_FLOAT32);

    // init booster
    LGBM_BoosterCreate(trainData, training_params, &booster);
    // train
    for (int i = 0; i < stoi(training_params["num_iterations"]); i++) {
        int isFinished;
        LGBM_BoosterUpdateOneIter(booster, &isFinished);
        if (isFinished) {
            break;
        }
    }

    int64_t len;
    vector<double> result(training_data->indptr.size() - 1);
    LGBM_BoosterPredictForCSR(booster,
                              static_cast<void *>(training_data->indptr.data()),
                              C_API_DTYPE_INT32,
                              training_data->indices.data(),
                              static_cast<void *>(training_data->data.data()),
                              C_API_DTYPE_FLOAT64,
                              training_data->indptr.size(),
                              training_data->data.size(),
                              n_feature,  //remove future t
                              C_API_PREDICT_NORMAL,
                              0,
                              training_params,
                              &len,
                              result.data());


    double se = 0;
    for (int i = 0; i < result.size(); ++i) {
        auto diff = result[i] - training_data->labels[i];
        se += diff * diff;
    }
    training_loss = training_loss * 0.99 + se / batch_size * 0.01;

    LGBM_DatasetFree(trainData);
    training_time = 0.95 * training_time +
                    0.05 * chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - timeBegin).count();
}

void CedLRBCache::sample() {
    // start sampling once cache filled up

    // sample random index
    // see how many are in cache and out of cache
    size_t index_to_cache = _distribution(_generator);
    size_t in_cache_size = (size_t)(in_cache_metas.size());
    size_t out_cache_size = (size_t)(out_cache_metas.size());

    // create bernoulli distribution from in/(in + out)
    // sample from this distribution
    double p = (double)in_cache_size / (double)(in_cache_size + out_cache_size);
    bernoulli_distribution _b_distribution(p);
    bool from_in = _b_distribution(_generator);

    // samplefrom in cache or out of cache metas based on this
    Meta *meta;    
    if (from_in) {
        index_to_cache %= in_cache_size;
        meta = &in_cache_metas[index_to_cache];
    } else {
        index_to_cache %= out_cache_size;
        meta = &out_cache_metas[index_to_cache];
    }
    Meta &m = *meta;
    m.emplace_sample(current_seq);
}


void CedLRBCache::update_stat_periodic() {
    float percent_beyond;
    if (0 == obj_distribution[0] && 0 == obj_distribution[1]) {
        percent_beyond = 0;
    } else {
        percent_beyond = static_cast<float>(obj_distribution[1])/(obj_distribution[0] + obj_distribution[1]);
    }
    obj_distribution[0] = obj_distribution[1] = 0;
    segment_percent_beyond.emplace_back(percent_beyond);
    segment_n_retrain.emplace_back(n_retrain);
    segment_n_in.emplace_back(in_cache_metas.size());
    segment_n_out.emplace_back(out_cache_metas.size());

    float positive_example_ratio;
    if (0 == training_data_distribution[0] && 0 == training_data_distribution[1]) {
        positive_example_ratio = 0;
    } else {
        positive_example_ratio = static_cast<float>(training_data_distribution[1])/(training_data_distribution[0] + training_data_distribution[1]);
    }
    training_data_distribution[0] = training_data_distribution[1] = 0;
    segment_positive_example_ratio.emplace_back(positive_example_ratio);

    n_retrain = 0;
    cerr
            << "in/out metadata: " << in_cache_metas.size() << " / " << out_cache_metas.size() << endl
            //    cerr << "feature overhead: "<<feature_overhead<<endl;
            << "memory_window: " << memory_window << endl
//            << "percent_beyond: " << percent_beyond << endl
//            << "feature overhead per entry: " << static_cast<double>(feature_overhead) / key_map.size() << endl
//            //    cerr << "sample overhead: "<<sample_overhead<<endl;
//            << "sample overhead per entry: " << static_cast<double>(sample_overhead) / key_map.size() << endl
            << "n_training: " << training_data->labels.size() << endl
            //            << "training loss: " << training_loss << endl
            << "training_time: " << training_time << " ms" << endl
            << "inference_time: " << inference_time << " us" << endl;
    assert(in_cache_metas.size() + out_cache_metas.size() == key_map.size());
}


bool CedLRBCache::lookup(const SimpleRequest &req) {

    // goal here: look for request inside cache (both lru and lrb caches)
    // increment sequence
    // we're going to advance our memory window since we now have a new entry,
    // so we need to forget information associated with the last entry
    // in our cache
    bool ret = false; // default to it not being in cache
    current_seq++;
    forget();
    // now we have to lookup the new key
    auto new_key_it = key_map.find(req.id);
    // if it exists in our map of training datums
    if (new_key_it != key_map.end()) {
        // is it in in_cache or out_cache training info?
        unsigned int in_or_out = new_key_it->second.list_idx;
        // where is it in there
        unsigned int pos = new_key_it->second.list_pos;
        // grab it
        Meta *meta;
        if (in_or_out == 0) {
            meta = &in_cache_metas[pos]; 
        } else {
            meta = &out_cache_metas[pos]; 
        }
        // save this so it doesn't get updated
        unsigned long last_ts = meta->_past_timestamp;
        unsigned long forget_ts = meta->_past_timestamp % memory_window;

        // do the same training shit
        if (!meta->_sample_times.empty()) {
            // loop through all sample times in training item, update times and counts
            for (unsigned int &sample_time : meta->_sample_times) {
                // what is the future time it is accessed? do labeling
                unsigned int future_distance = current_seq - sample_time;
                training_data->emplace_back(*meta, sample_time, future_distance, meta->_key);
                training_data_distribution[1]++;
            }
            // if there are enough items in training set, then we can train
            unsigned long nb_items = training_data->labels.size();
            if (nb_items >= batch_size) {
                train();
                training_data->clear();
            }
            // now we can remove this request, evict!
            meta->_sample_times.clear();
            meta->_sample_times.shrink_to_fit();
        }
        meta->update(current_seq);

        // what if it's in out_cache? then we need to update the queue
        if (in_or_out == 1) {
            // remove old version
            negative_candidate_queue->erase(forget_ts);
            // new version has an updated timestamp for this key
            pair<unsigned long, unsigned long> _pair = {current_seq % memory_window, req.id};
            negative_candidate_queue->insert(_pair);
        } else {
            // otherwise we just need to put it in lru queue
            InCacheMeta *p = (InCacheMeta *)meta;
            p->p_last_request = in_cache_lru_queue.re_request(p->p_last_request);
        }
        // intuition: if req was in second queue, then it's in cache
        // since it's inside in_cache_metas
        if (pos == 1)
            ret = true;
    }
    // get candidates now
    if (is_sampling) {
        sample();
    }
    return ret;
}


void CedLRBCache::forget() {
    // happens when outside memory_window
    // get item from candidates
    auto it = negative_candidate_queue->find(current_seq % memory_window);
    // if there is an item
    if (it != negative_candidate_queue->end()) {
        // get the key, and if it's mature, do the training shit again
        unsigned long _key = it->second;
        unsigned int pos = key_map.find(_key)->second.list_pos;
        Meta *meta = &out_cache_metas[pos];
        // is it part of our training data?
        if (!meta->_sample_times.empty()) {
            // what is the future time it is accessed?
            // since it hasn't been accessed again in our memory_window, just say twice
            unsigned int future_distance = memory_window * 2; 
            // loop through all sample times in training item, update times and counts
            for (unsigned int &sample_time : meta->_sample_times) {
                training_data->emplace_back(*meta, sample_time, future_distance, meta->_key);
                training_data_distribution[0]++;
            }
            // if there are enough items in training set, then we can train
            unsigned long nb_items = training_data->labels.size();
            if (nb_items >= batch_size) {
                train();
                training_data->clear();
            }
            // now we can remove this request, evict!
            meta->_sample_times.clear();
            meta->_sample_times.shrink_to_fit();
        }
        remove_from_outcache_metas(*meta, pos, _key);
    }
}

void CedLRBCache::admit(const SimpleRequest &req) {

    // check if the object fits in cache

    // check if object is already in cache
    //     if it's not, then add it
    //     if it is, then do some shit with the metadata

    // now it's admitted. if cache not full, then just return
    // if cachhe is full, then we can start doing lrb (sampling true)
    // if cache is full, then remove excess

    if (req.size > _cacheSize) {
        LOG("L", _cacheSize, req.id, size);
        return;
    }

    auto it = key_map.find(req.id);

    if (it == key_map.end()) {
        // kmentry has first int as which list it's it (in or out metas)
        // and second entry is where in that list it is (should insert at back)
        KeyMapEntryT kmentry = {0, (unsigned int)in_cache_metas.size()};
        // key_map is map from id to entry information
        pair<unsigned long, KeyMapEntryT> p = {req.id, kmentry};
        key_map.insert(p);

        // grab it from our lru cache, which is where it's stored if we haven't
        // seen it before
        auto lru_iterator = in_cache_lru_queue.request(req.id);
        // now add it to cache
        Meta new_req = Meta(req.id, req.size, current_seq, req.extra_features);
        in_cache_metas.emplace_back(new_req, lru_iterator);
    } else {
        // it's already in our second cache (lru), and we want to admit to lrb
        // that means to update metadata and update hash table

        unsigned int lrb_pos = in_cache_metas.size(); // inserting at back of lrb cache
        // where in the second (lru) cache is it?
        unsigned int lru_pos = it->second.list_pos;
        Meta *meta = &out_cache_metas[lru_pos];
        // can now forget this timestamp
        unsigned int forget_ts = meta->_past_timestamp % memory_window;
        negative_candidate_queue->erase(forget_ts);
        // place element in lrb cache now
        auto lru_iterator = in_cache_lru_queue.request(req.id);
        in_cache_metas.emplace_back(*meta, lru_iterator);

        // now need to decrement second (lru) cache size, and update cache entries
        unsigned int last = out_cache_metas.size() - 1;
        // if the entry we admitted wasn't the last entry of our second
        // cache then we need to fix that entry
        if (lru_pos != last) {
            out_cache_metas[lru_pos] = out_cache_metas[last];
            auto sit = key_map.find(out_cache_metas[last]._key);
            sit->second.list_pos = lru_pos;
        }
        // now can remove last entry of second cache
        out_cache_metas.pop_back(); 
        // and update this entry to lrb cache at this position
        it->second = {0, lrb_pos}; 
        // and increment size of our cache
    }
    _currentSize += req.size; 
    if (_currentSize > _cacheSize)
        is_sampling = true;

    while (_currentSize > _cacheSize)
        evict();
}


pair<uint64_t, uint32_t> CedLRBCache::rank() {
    // now we rank entries for eviction
    // check candidate for eviction
    // if we haven't gotten enough training data, then just default
    unsigned long key = in_cache_lru_queue.dq.back();
    auto it = key_map.find(key);
    unsigned int pos = it->second.list_pos;
    InCacheMeta *meta = &in_cache_metas[pos];
    // are we past the memory window?
    if (current_seq - meta->_past_timestamp > memory_window) {
        // need to tractk stats for objects ranked
        if (booster) {
            obj_distribution[1]++;
        }
        return {meta->_key, pos};
    }

    // left rest of ranking function as is, couldn't figure what these data
    // structures were for
    int32_t indptr[sample_rate + 1];
    indptr[0] = 0;
    int32_t indices[sample_rate * n_feature];
    double data[sample_rate * n_feature];
    int32_t past_timestamps[sample_rate];
    uint32_t sizes[sample_rate];

    unordered_set<uint64_t> key_set;
    uint64_t keys[sample_rate];
    uint32_t poses[sample_rate];
    //next_past_timestamp, next_size = next_indptr - 1

    unsigned int idx_feature = 0;
    unsigned int idx_row = 0;

    auto n_new_sample = sample_rate - idx_row;
    while (idx_row != sample_rate) {
        uint32_t pos = _distribution(_generator) % in_cache_metas.size();
        auto &meta = in_cache_metas[pos];
        if (key_set.find(meta._key) != key_set.end()) {
            continue;
        } else {
            key_set.insert(meta._key);
        }

        keys[idx_row] = meta._key;
        poses[idx_row] = pos;
        //fill in past_interval
        indices[idx_feature] = 0;
        data[idx_feature++] = current_seq - meta._past_timestamp;
        past_timestamps[idx_row] = meta._past_timestamp;

        uint8_t j = 0;
        uint32_t this_past_distance = 0;
        uint8_t n_within = 0;
        if (meta._extra) {
            for (j = 0; j < meta._extra->_past_distance_idx && j < max_n_past_distances; ++j) {
                uint8_t past_distance_idx = (meta._extra->_past_distance_idx - 1 - j) % max_n_past_distances;
                uint32_t &past_distance = meta._extra->_past_distances[past_distance_idx];
                this_past_distance += past_distance;
                indices[idx_feature] = j + 1;
                data[idx_feature++] = past_distance;
                if (this_past_distance < memory_window) {
                    ++n_within;
                }
            }
        }

        indices[idx_feature] = max_n_past_timestamps;
        data[idx_feature++] = meta._size;
        sizes[idx_row] = meta._size;

        for (uint k = 0; k < n_extra_fields; ++k) {
            indices[idx_feature] = max_n_past_timestamps + k + 1;
            data[idx_feature++] = meta._extra_features[k];
        }

        indices[idx_feature] = max_n_past_timestamps + n_extra_fields + 1;
        data[idx_feature++] = n_within;

        for (uint8_t k = 0; k < n_edc_feature; ++k) {
            indices[idx_feature] = max_n_past_timestamps + n_extra_fields + 2 + k;
            uint32_t _distance_idx = min(uint32_t(current_seq - meta._past_timestamp) / edc_windows[k],
                                         max_hash_edc_idx);
            if (meta._extra)
                data[idx_feature++] = meta._extra->_edc[k] * hash_edc[_distance_idx];
            else
                data[idx_feature++] = hash_edc[_distance_idx];
        }
        //remove future t
        indptr[++idx_row] = idx_feature;
    }

    int64_t len;
    double scores[sample_rate];
    system_clock::time_point timeBegin;
    //sample to measure inference time
    if (!(current_seq % 10000))
        timeBegin = chrono::system_clock::now();
    LGBM_BoosterPredictForCSR(booster,
                              static_cast<void *>(indptr),
                              C_API_DTYPE_INT32,
                              indices,
                              static_cast<void *>(data),
                              C_API_DTYPE_FLOAT64,
                              idx_row + 1,
                              idx_feature,
                              n_feature,  //remove future t
                              C_API_PREDICT_NORMAL,
                              0,
                              inference_params,
                              &len,
                              scores);
    if (!(current_seq % 10000))
        inference_time = 0.95 * inference_time +
                         0.05 *
                         chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - timeBegin).count();
    for (int i = sample_rate - n_new_sample; i < sample_rate; ++i) {
        //only monitor at the end of change interval
        if (scores[i] >= log1p(memory_window)) {
            ++obj_distribution[1];
        } else {
            ++obj_distribution[0];
        }
    }

    if (objective == object_miss_ratio) {
        for (uint32_t i = 0; i < sample_rate; ++i)
            scores[i] *= sizes[i];
    }

    vector<int> index(sample_rate, 0);
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    sort(index.begin(), index.end(),
         [&](const int &a, const int &b) {
             return (scores[a] > scores[b]);
         }
    );

    return {keys[index[0]], poses[index[0]]};
}

void CedLRBCache::evict() {

    // get key and position of highest rank to evict
    // get that request
    // if current_seq - past_timestamp > memory_window
    // if it's in lru cache
    // if it is "mature" i.e. we've seen it before and can create  training data
    // get future access time
    // update training data
    // train if possible (batch size large enough)
    // clear data
    // erase that shit from cache
    // fix the tail
    // then if it's not in lru cache
    // take it out
    

    // get key and position of highest rank to evict
    pair<unsigned long, unsigned int> p = rank();
    unsigned long &key = p.first;
    unsigned int &pos = p.second;
    // get that request
    InCacheMeta *meta = &in_cache_metas[pos];
    // check if it's in lru cache
    if(current_seq - meta->_past_timestamp > memory_window) {
        // is it part of our training data?
        if (!meta->_sample_times.empty()) {
            // what is the future time it is accessed?
            unsigned int future_distance = current_seq - meta->_past_timestamp + memory_window;
            // loop through all sample times in training item, update times and counts
            for (unsigned int &sample_time : meta->_sample_times) {
                training_data->emplace_back(*meta, sample_time, future_distance, meta->_key);
                training_data_distribution[0]++;
            }
            // if there are enough items in training set, then we can train
            unsigned long nb_items = training_data->labels.size();
            if (nb_items >= batch_size) {
                train();
                training_data->clear();
            }
            // now we can remove this request, evict!
            meta->_sample_times.clear();
            meta->_sample_times.shrink_to_fit();

            in_cache_lru_queue.dq.erase(meta->p_last_request);
            // update iterator to end of cache
            meta->p_last_request = in_cache_lru_queue.dq.end();
            meta->free();
            _currentSize -= meta->_size;
            key_map.erase(key);

            // now need to decrement first (lrb) cache size, and update cache entries
            unsigned int last = in_cache_metas.size() - 1;
            // if the entry we are to evict wasn't the last entry of our second
            // cache then we need to fix that entry
            if (pos != last) {
                in_cache_metas[pos] = in_cache_metas[last];
                auto sit = key_map.find(in_cache_metas[last]._key);
                sit->second.list_pos = pos;
            }
            // now evict
            in_cache_metas.pop_back();
            n_force_eviction++;
        } else {
            in_cache_lru_queue.dq.erase(meta->p_last_request);
            meta->p_last_request = in_cache_lru_queue.dq.end();
            _currentSize -= meta->_size;
            unsigned long actual_timestamp = meta->_past_timestamp % memory_window;
            unsigned long new_key = meta->_key;
            pair<unsigned long, unsigned long> _pair = {actual_timestamp, new_key};
            negative_candidate_queue->insert(_pair);
            
            // now need to decrement second (lru) cache size, and update cache entries
            unsigned int _new = out_cache_metas.size() - 1;
            out_cache_metas.emplace_back(in_cache_metas[_new]);
            // now need to decrement first (lrb) cache size, and update cache entries
            unsigned int last = in_cache_metas.size() - 1;
            // if the entry we are to evict wasn't the last entry of our second
            // cache then we need to fix that entry
            if (pos != last) {
                in_cache_metas[pos] = in_cache_metas[last];
                auto sit = key_map.find(in_cache_metas[last]._key);
                sit->second.list_pos = pos;
            }
            in_cache_metas.pop_back();
            key_map.find(key)->second = {1, _new};
        } 
    }
}

void CedLRBCache::remove_from_outcache_metas(Meta &meta, unsigned int &pos, const uint64_t &key) {
    //free the actual content
    meta.free();
    //TODO: can add a function to delete from a queue with (key, pos)
    //evict
    uint32_t tail_pos = out_cache_metas.size() - 1;
    if (pos != tail_pos) {
        //swap tail
        out_cache_metas[pos] = out_cache_metas[tail_pos];
        key_map.find(out_cache_metas[tail_pos]._key)->second.list_pos = pos;
    }
    out_cache_metas.pop_back();
    key_map.erase(key);
    negative_candidate_queue->erase(current_seq % memory_window);
}

