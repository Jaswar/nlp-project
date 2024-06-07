import argparse
import os
import json
from tqdm import tqdm

from prompts import Prompt_Loader
from utils import OpenAIModel

class Reasoning_Program_Generator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.num_programs_per_example = args.num_programs_per_example
        self.checkpoint_file = args.checkpoint_file

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.prompt_loader = Prompt_Loader()

    def update_results(self, sample, generated_text):
        program_list = [operation.strip() for operation in generated_text.split('\n')]
        # programs = [program_list]
        self.result_dict[sample['id']]['predicted_programs'].append(program_list)

    def batch_generate_programs(self):
        batch_size = 1  # other batch sizes not supported
        # create output_dir
        self.result_dict = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # load dataset
        with open(os.path.join(self.data_path, 'NumTemp-E9C0', 'data', 'raw_data', 'val_claims_quantemp.json'), 'r') as f:
            raw_dataset = json.load(f)
        
        raw_dataset = raw_dataset if self.args.num_eval_samples < 0 else raw_dataset[:self.args.num_eval_samples]
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name} dev set.")

        # generate programs
        temperature = 0.0 if self.num_programs_per_example == 1 else 0.7
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        
        # initialize empty results
        result_dict = {}

        for idx, sample in enumerate(raw_dataset):
            sample['id'] = idx
            result = {'idx': idx,
                        'id': sample['id'],
                        'claim': sample['claim'],
                        'gold': sample['label'],
                        'predicted_programs': []}
            result_dict[idx] = result

        if self.checkpoint_file is not None:
            with open(self.checkpoint_file, 'r') as f:
                o = json.load(f)
                for x in o:
                    result_dict[x['id']] = x

        self.result_dict = result_dict

        # for each iteration
        for iteration in range(self.num_programs_per_example):
            # print(f"Generating programs for iteration {iteration + 1}...")
            # for each chunk
            for chunk in tqdm(dataset_chunks):
                if len(result_dict[chunk[0]['id']]['predicted_programs']) > 0:
                    continue
                # create prompt
                full_prompts = [self.prompt_loader.prompt_construction(example['claim'], self.dataset_name) for example in chunk]
                try:
                    batch_outputs = self.openai_api.batch_generate(full_prompts, temperature)
                    # create output
                    for sample, output in zip(chunk, batch_outputs):
                        self.update_results(sample, output)
                except:
                    # generate one by one if batch generation fails
                    for sample, full_prompt in zip(chunk, full_prompts):
                        try:
                            output = self.openai_api.generate(full_prompt, temperature)
                            self.update_results(sample, output)
                        except Exception as e:
                            print('Error in generating reasoning programs for example: ', sample['id'], e)

                # create outputs
                outputs = []
                for key in result_dict:
                    outputs.append(result_dict[key])
                sorted_outputs = sorted(outputs, key=lambda x: x['idx'])

                # save outputs
                with open(os.path.join(self.save_path, f'{self.dataset_name}_N={self.num_programs_per_example}_{self.model_name}_programs.json'), 'w') as f:
                    json.dump(sorted_outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='HOVER', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--num_eval_samples', default=-1, type=int)
    parser.add_argument('--num_programs_per_example', default=1, type=int)
    parser.add_argument('--save_path', default = './results/programs', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--stop_words', type=str, default='# The claim is')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--checkpoint_file', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    generator = Reasoning_Program_Generator(args)
    generator.batch_generate_programs()