# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from code.harness.harness_llm_py.llm_server import LLMDataset


class LlamaDataset(LLMDataset):
    FILES = [
        'input_ids_padded.npy',
        'input_lens.npy',
    ]

    def post_init(self):
        # truncate inputs to proper lengths
        self.input_table = [
            input_ids[:input_len].reshape(-1).tolist()
            for input_ids, input_len in zip(self.input_ids_padded, self.input_lens)
        ]

        self.logger.verbose("Completed pre-processing input tokens. Ready for inference.")

    def get_input_tokens(self, sample_indices):
        return [
            self.input_table[sample_index]
            for sample_index in sample_indices
        ]

    def __len__(self):
        return len(self.input_table)
