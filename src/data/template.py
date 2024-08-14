from jinja2 import Template 
from typing import List, Dict, Union

def template_cls(template_name):
    return {
        'language': LanguageTemplate,
        'table': TableTemplate,
        'anonymous_table': AnonymousTableTemplate
    }[template_name]

class TabularTemplate:
    
    def __init__(self, tokenizer, config, meta_info, task_info):
        self.tokenizer = tokenizer
        self.config = config
        self.meta_info = meta_info
        self.task_info = task_info
        
        # Customized variables
        self.mask_token = '<MASK>'
        self.mask_target = False
        self.answer_decimal = 0
        self.max_decimal = self.config.max_decimal
    
    def load_task_info(self, task_info):
        self.label_column = task_info.label_column
        
        # Load prompt
        self.prompt = self.meta_info['task_info'][self.label_column]['prompt']
        self.role_prompt = self.prompt['role_prompt']
        self.task_prompt = self.prompt['task_prompt']
        
        # Define answer format
        if task_info.task_type == 'classification':
            self.answer_prompt = self.prompt['answer_prompt']
        else:
            self.answer_decimal = self.meta_info['feature_info'][self.label_column]['decimal']
            self.answer_prompt = ""
    
    def format_number(self, num: Union[int, float], decimal: Union[int, str] = 0) -> str:
        if decimal == 0:
            return str(int(num))
        if decimal == 'sci':
            return f'{num:.2e}'
        return str(round(float(num), min(decimal, self.max_decimal)))
    
    def get_answer_prompt(self, answer: Union[int, float], is_context: bool, is_mask: bool):
        # Mask target sample answer
        if is_mask:
            return self.mask_token

        # Format answer
        answer = self.format_number(answer, self.answer_decimal)
        if is_context:
            return answer
        else:
            return f'{self.tokenizer.answer_begin_token}{answer}{self.tokenizer.answer_end_token}'
    
    def get_data_prompt(
        self,
        sample: Dict,
        suffix: str = '',
        is_context: bool = False
    ):
        feature_prompt = sample[f'feature_prompt{suffix}']
        answer_prompt = self.get_answer_prompt(
            sample['label_tabular'],
            is_context=is_context,
            is_mask=(self.mask_target and not is_context)
        )
        data_prompt = self.data_template(feature_prompt, answer_prompt)
        return data_prompt
    
    def get_data_prompt_with_context(self, sample: Dict, context_samples: List[Dict], suffix: str = ''):
        data_prompt_list = []
        
        for context_sample in context_samples:
            context_data_prompt = self.get_data_prompt(context_sample, suffix, is_context=True)
            data_prompt_list.append(context_data_prompt)
        
        target_data_prompt = self.get_data_prompt(sample, suffix, is_context=False)
        data_prompt_list.append(target_data_prompt)
        
        data_prompt = '\n'.join(data_prompt_list)
        return data_prompt

    def generate_full_prompt(self, sample: Dict, context_samples: List[Dict] = None):
        if context_samples is None:
            sample['prompt'] = self.zero_shot_template(
                sample, self.get_data_prompt(sample)
            )
        else:
            sample['prompt'] = self.in_context_learning_template(
                sample, self.get_data_prompt_with_context(sample, context_samples)
            )
        return sample
    
    def generate_feature_prompt(self, sample: Dict):
        raise NotImplementedError
    
    def data_template(self, feature_prompt: str, answer_prompt: str):
        raise NotImplementedError
    
    def zero_shot_template(self, sample: Dict, data_prompt: str):
        raise NotImplementedError
    
    def in_context_learning_template(self, sample: Dict, data_prompt: str):
        raise NotImplementedError


class LanguageTemplate(TabularTemplate):
    
    def __init__(self, tokenizer, config, meta_info, task_info):
        super().__init__(tokenizer, config, meta_info, task_info)
        # Task-specific variables
        self.load_task_info(task_info)
    
    def get_feature_notes(self, sample):
        notes = []
        # Add numerical feature notes
        num_cols = self.meta_info['basic_info']['num_features']
        for i, mask in enumerate(sample['num_feats_mask']):
            if mask == 0:
                continue
            feat_name = num_cols[i]
            feat_desc = self.meta_info['feature_info'][feat_name]['description'].rstrip('.')
            decimal = self.meta_info['feature_info'][feat_name]['decimal']
            note = f'{feat_desc} is{self.tokenizer.num_begin_token}' \
                    f'{self.format_number(sample["num_feats"][i], decimal)}' \
                    f'{self.tokenizer.num_end_token}.'
            notes.append(note)
        
        # Add categorical feature notes
        cat_cols = self.meta_info['basic_info']['cat_features'] + self.meta_info['basic_info']['other_features']
        for i, mask in enumerate(sample['cat_feats_mask']):
            if mask == 0:
                continue
            feat_name = cat_cols[i]
            feat_desc = self.meta_info['feature_info'][feat_name]['description'].rstrip('.')
            feat_value = sample["cat_feats"][i].rstrip('.')
            if 'value_dict' in self.meta_info['feature_info'][feat_name] and \
                feat_value in self.meta_info['feature_info'][feat_name]['value_dict']:
                
                feat_value_desc = self.meta_info['feature_info'][feat_name]['value_dict'][feat_value].rstrip('.')
                # [subject] + [verb] + [object]
                if len(feat_value_desc.split(' ')) >= 3:
                    note = f'{feat_value_desc}.'
                else:
                    note = f'{feat_desc}: {feat_value_desc}.'
            else:
                note = f'{feat_desc}: {feat_value}.'
            notes.append(note)
        
        return notes
    
    def generate_feature_prompt(self, sample):
        notes = self.get_feature_notes(sample)
        sample['feature_prompt'] = ' '.join(notes)
        return sample
    
    def get_answer_prompt(self, answer: Union[int, float], is_context: bool, is_mask: bool):
        # Mask target sample answer
        if is_mask:
            return self.mask_token

        # Format answer
        answer = self.format_number(answer, self.answer_decimal)
        if is_context:
            # To align the answer format with/without the special tokens
            return f' {answer}'
        else:
            # Tokenize will add an extra space after each special token
            return f'{self.tokenizer.answer_begin_token}{answer}{self.tokenizer.answer_end_token}'
    
    def data_template(self, feature_prompt, answer_prompt):
        data_template = f'Features: {feature_prompt}\nAnswer:{answer_prompt}'
        return data_template
    
    def zero_shot_template(self, sample: Dict, data_prompt: str):
        # Create a template  
        template = Template(  
            "{{ role_prompt }}\n"
            "{{ task_prompt }}\n"
            "{{ answer_prompt }}\n"
            "{{ data_prompt }}"
        )
        
        # Render the template with actual data  
        rendered_template = template.render({
            'role_prompt': self.role_prompt,
            'task_prompt': self.task_prompt,
            'answer_prompt': self.answer_prompt,
            'data_prompt': data_prompt
        })
        return rendered_template
    
    def in_context_learning_template(self, sample: Dict, data_prompt: str):
        # Create a template  
        template = Template(
            "{{ role_prompt }}\n"
            "{{ task_prompt }}\n"
            "{{ answer_prompt }}\n"
            "I will supply multiple instances with features and the corresponding label for your reference.\n"
            "{{ data_prompt }}"
        )
        
        # Render the template with actual data  
        rendered_template = template.render({
            'role_prompt': self.role_prompt,
            'task_prompt': self.task_prompt,
            'answer_prompt': self.answer_prompt,
            'data_prompt': data_prompt
        })
        return rendered_template
      

class TableTemplate(TabularTemplate):
    
    def __init__(self, tokenizer, config, meta_info, task_info):
        super().__init__(tokenizer, config, meta_info, task_info)
        self.missing_value_prompt = 'nan'
        self.sep_token = '|'
        self.mask_target = True
        if not config.load_synthetic:
            # Task-specific variables
            self.load_task_info(task_info)
            self.table_template = self.table_description_template()
    
    def generate_feature_prompt(self, sample):
        cat_feats = []
        for i, mask in enumerate(sample['cat_feats_mask']):
            cat_feats.append(sample['cat_feats'][i] if mask == 1 else self.missing_value_prompt)
        
        num_feats = []
        for i, mask in enumerate(sample['num_feats_mask']):
            feat_name = self.meta_info['basic_info']['num_features'][i]
            decimal = self.meta_info['feature_info'][feat_name]['decimal']
            num_feats.append(
                f'{self.tokenizer.num_begin_token}' \
                f'{self.format_number(sample["num_feats"][i], decimal)}' \
                f'{self.tokenizer.num_end_token}'
                    if mask == 1 else self.missing_value_prompt
            )

        sample['feature_prompt'] = self.sep_token.join(num_feats + cat_feats)
        return sample
    
    def data_template(self, feature_prompt: str, answer_prompt: str):
        data_template = f'{self.sep_token}{feature_prompt}{self.sep_token}{answer_prompt}{self.sep_token}'
        return data_template
    
    def table_description_template(self):
        table_template = Template(
            "Please refer to the table below for detailed descriptions of the features and label:\n"
            "--- feature description ---\n"
            "{% for feature in features %}"
            "{{ feature.name }}: {{ feature.description }}\n"
            "{% endfor %}"
            "--- label description --- \n"
            "{{ label.name }}: {{ label.description }}\n"
            "--- data ---\n"
            "{{ column_header }}"
        )
        
        # Feature description
        feat_names = self.task_info.num_features + self.task_info.cat_features
        features = [
            {
                'name': feat,
                'description': self.meta_info['feature_info'][feat]['description']
            } for feat in feat_names
        ]
        
        # Label description
        label = {
            'name': self.label_column,
            'description': self.meta_info['feature_info'][self.label_column]['description']
        }
        
        # Column header
        column_header = self.sep_token.join(feat_names + [self.label_column])
        column_header = f'{self.sep_token}{column_header}{self.sep_token}'
        
        rendered_table_template = table_template.render(features=features, label=label, column_header=column_header)
        return rendered_table_template
    
    def zero_shot_template(self, sample: Dict, data_prompt: str):
        # Create a template  
        template = Template(  
            "{{ role_prompt }}\n"
            "{{ task_prompt }}\n"
            "{{ table_template }}\n"
            "{{ data_prompt }}\n"
            "Please predict the {{ mask_token }} {{ label_column }}. {{ answer_prompt }}\n"
            "Answer:{{ target_answer }}"
        )
        
        # Render the template with actual data  
        rendered_template = template.render({  
            'role_prompt': self.role_prompt,
            'task_prompt': self.task_prompt,
            'table_template': self.table_template,
            'data_prompt': data_prompt,
            'mask_token': self.mask_token,
            'label_column': self.label_column,
            'answer_prompt': self.answer_prompt,
            'target_answer': self.get_answer_prompt(sample['label_tabular'], is_context=False, is_mask=False)
        })
        return rendered_template
    
    def in_context_learning_template(self, sample: Dict, data_prompt: str):
        # Create a template  
        template = Template(
            "{{ role_prompt }}\n"
            "{{ task_prompt }}\n"
            "I will supply multiple instances with features and the corresponding label for your reference.\n"
            "{{ table_template }}\n"
            "{{ data_prompt }}\n"
            "Please use the supplied data to predict the {{ mask_token }} {{ label_column }}. {{ answer_prompt }}\n"
            "Answer:{{ target_answer }}"
        )
        
        # Render the template with actual data
        rendered_template = template.render({
            'role_prompt': self.role_prompt,
            'task_prompt': self.task_prompt,
            'table_template': self.table_template,
            'data_prompt': data_prompt,
            'mask_token': self.mask_token,
            'label_column': self.label_column,
            'answer_prompt': self.answer_prompt,
            'target_answer': self.get_answer_prompt(sample['label_tabular'], is_context=False, is_mask=False)
        })
        return rendered_template

class AnonymousTableTemplate(TableTemplate):
    def __init__(self, tokenizer, config, meta_info, task_info):
        super().__init__(tokenizer, config, meta_info, task_info)
    
    def zero_shot_template(self, sample: Dict, data_prompt: str):
        # Create a template  
        template = Template(  
            "The following is a table with features in the first few columns and the label in the last column.\n"
            "{{ data_prompt }}\n"
            "Please predict the {{ mask_token }} label.\n"
            "Answer:{{ target_answer }}"
        )
        
        # Render the template with actual data  
        rendered_template = template.render({  
            'data_prompt': data_prompt,
            'mask_token': self.mask_token,
            'target_answer': self.get_answer_prompt(sample['label_tabular'], is_context=False, is_mask=False)
        })
        return rendered_template
    
    def in_context_learning_template(self, sample: Dict, data_prompt: str):
        # Create a template  
        template = Template(  
            "The following is a table with features in the first few columns and the label in the last column.\n"
            "I will supply multiple instances with features and the corresponding label for your reference.\n"
            "{{ data_prompt }}\n"
            "Please use the supplied data to predict the {{ mask_token }} label.\n"
            "Answer:{{ target_answer }}"
        )
        
        # Render the template with actual data  
        rendered_template = template.render({  
            'data_prompt': data_prompt,
            'mask_token': self.mask_token,
            'target_answer': self.get_answer_prompt(sample['label_tabular'], is_context=False, is_mask=False)
        })
        return rendered_template