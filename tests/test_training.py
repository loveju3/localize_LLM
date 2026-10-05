import unittest
from training.lora import TrainConfig, encode_sample, read_split


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'abc' if kwargs['add_generation_prompt'] else 'abcde'

    def __call__(self, text, **kwargs):
        return {'input_ids':[1,2,3,4,5], 'offset_mapping':[(0,1),(1,2),(2,3),(3,4),(4,5)]}


class TrainingTests(unittest.TestCase):
    def test_completion_only_mask_and_fail_closed_on_template_drift(self):
        row = {'messages':[{'role':'system','content':'rules'},
                           {'role':'user','content':'question'}, {'role':'assistant','content':'answer'}]}
        result = encode_sample(Tokenizer(), row, 10)
        self.assertEqual(result['labels'], [-100, -100, -100, 4, 5])
        with self.assertRaises(ValueError):
            encode_sample(Tokenizer(), row, 4)
        class Changed(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return 'x' if kwargs['add_generation_prompt'] else 'abc'
        with self.assertRaisesRegex(ValueError, 'prefix mismatch'):
            encode_sample(Changed(), row, 10)

    def test_config_bounds_and_no_test_split_training(self):
        TrainConfig('valid-run').validate()
        for kwargs in [{'run_id':'../escape'}, {'rank':0}, {'max_steps':0},
                       {'learning_rate':-1}, {'revision':'main'}, {'max_length':99999}]:
            with self.assertRaises(ValueError):
                TrainConfig(**({'run_id':'run'} | kwargs)).validate()
        self.assertEqual(len(read_split('datasets/lol/v1/train.jsonl')), 60)
        self.assertEqual(len(read_split('datasets/lol/v1/validation.jsonl')), 20)
