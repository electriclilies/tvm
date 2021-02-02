# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

class DatasetManager():
    def __init__(self):
        raise NotImplementedError

    def get_next_batch(self):
        raise NotImplementedError

    def num_batches(self):
        raise NotImplementedError

    def is_empty(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class TFDatasetManager(DatasetManager):
    def __init__(self, tf_dataset, batch_size, n_batches):
        self.idx = 0
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.tf_dataset = tf_dataset
        self.tf_iter = iter(self.tf_dataset)
    
    # Returns the next batch of data
    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        self.idx += 1
        
        data, label = next(self.tf_iter)

        return [data.numpy()], label.numpy()
    
    def num_batches(self):
        return self.n_batches

    def is_empty(self):
        return self.idx >= self.n_batches

    # TODO: either reset automatically at the end of get batch or have a is_empty util
    def reset(self):
        self.tf_iter = iter(self.tf_dataset)
        self.idx = 0
