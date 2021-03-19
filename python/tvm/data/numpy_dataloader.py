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

from tvm.data import DataLoader


class NumpyDataLoader(DataLoader):
    """DataLoader wrapping a dataset stored in a numpy array. Intended for use on keras datasets.
    See https://www.tensorflow.org/tutorials/images/cnn for an example of how to construct the keras
    dataset to pass into this class.

    Parameters
    ----------
    numpy_dataset : ndarray
        An ndarray containing all the datapoints. For now, the ndarray must be in NCHW format, with the
        N equaling the total number of inputs in the dataset.

    numpy_labels : ndarray
        An ndarray containing all the labels. The length of the ndarray must be the same as the N dimension
        of numpy_dataset.

    batch_size : int, optional
        Number of datapoints to put in a batch.

    num_batches : int, optional
        Number of batches to iterate through. If None, num_batches will be the maximum number of batches
        given the batch size. num_batches * batch_size must be less than or equal to the N dimension of
        the numpy_dataset.

    layout : str, optional
        String representing the layout the numpy_dataset is in. Currently we only support NCHW as the layout.
    """

    def __init__(self, numpy_data, numpy_labels, batch_size=1, num_batches=None, layout="NCHW"):
        self.idx = 0
        self.numpy_data = numpy_data
        self.numpy_labels = numpy_labels
        assert layout == "NCHW", "NumpyDataLoader currently only supports NCHW layout. "
        assert (
            self.numpy_data.shape[0] == self.numpy_labels.shape[0]
        ), "First dimension of data and label arrays must match."
        assert (
            self.numpy_data.shape[0] >= batch_size
        ), "Batch size too large. You must provide enough data points for at least one batch."
        self.batch_size = batch_size
        if num_batches is None:
            self.num_batches = numpy_data.shape[0] // self.batch_size
        else:
            assert num_batches * batch_size <= numpy_data.shape[0]
            self.num_batches = num_batches

    def get_next_batch(self):
        """Returns the next batch from the tensorflow dataset and its labels.

        Returns
        -------
        data : List of ndarray
            List containing the data from the numpy dataset.

        label : List of int
            List of the labels from the numpy dataset. Length is equal to batch size.
        """
        if self.is_empty():
            raise IndexError
        batched_data = self.numpy_data[
            self.idx * self.batch_size : (self.idx + 1) * self.batch_size
        ]
        batched_label = self.numpy_labels[
            self.idx * self.batch_size : (self.idx + 1) * self.batch_size
        ]
        self.idx += 1
        return [batched_data], batched_label

    def get_num_batches(self):
        """Gets the number of batches.
        Returns
        -------
        num_batches : int
            The total number of batches in the DataLoader.
        """
        return self.num_batches

    def get_batch_size(self):
        """Gets the batch size.

        Returns
        -------
        batch_size : int
            The size of the batch returned by the DataLoader.
        """
        return self.batch_size

    def is_empty(self):
        """Checks whether the DataLoader has any batches left.

        Returns
        -------
        is_empty : bool
            Whether there are any batches left in the DataLoader.
        """
        return self.idx >= self.num_batches

    def reset(self):
        """Resets the DataLoader to the beginning."""
        self.idx = 0
