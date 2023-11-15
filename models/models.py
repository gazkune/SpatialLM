import math
import time
import token
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertLayer, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertPreTrainingHeads, BertConfig
import torch.nn.functional as F

from copy import deepcopy

# Anderren komentarioak:
# Hemen daukazu kode snipettak. MMBERTForSequenceClassification klasea 
# models.py fitxategian dago, BertVisualModel eta 
# BertEmbeddingsWithVisualEmbeddings klaseekin batera (beharrezkoak 
# direnak). transformers.py fitxategian MMBERTForSequenceClassification 
# eredua inizializatzen dut load_model() funtzioan eta forward_pass() 
# funtzioaren bukaeran deitzen zaio inferentzia egiteko.


# Original code from https://github.com/uclanlp/visualbert by Liunian Harold Li


class BertEmbeddingsWithSpatialEmbedding(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and spatial embeddings (for object-attribute pairs).
    We will have a list of tokens (token ids). The first tokens (question) should have normal treatment.
    The second sequence of tokens (after [SEP]?) are object-attribute sequences. They should be treated differently:
    Obtain the word embedding. Add (projected) spatial embeddings to all tokens forming an object-attribute set. Add also token type embeddings? 
    """

    def __init__(self, config, spatial_embedding_dim=1024):
        super(BertEmbeddingsWithSpatialEmbedding, self).__init__()        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Below are specific for encoding object-wise features

        # Segment and position embedding for object-wise features
        self.token_type_embeddings_objects = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # NOTE: I think I don't need this line below (comment)
        #self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.projection = nn.Linear(spatial_embedding_dim, config.hidden_size)
        
        # self.special_intialize() # NOTE: Why is this commented? 
        
    # NOTE: As the line above is commented, I assume I don't need this. Don't touch it
    def special_intialize(self, method_type=0):
        # This is a bit unorthodox. The better way might be to add an initializer to AllenNLP. This function is used to
        # initialize the token_type_embeddings_visual and position_embedding_visual, just in case.
        self.token_type_embeddings_visual.weight = torch.nn.Parameter(deepcopy(self.token_type_embeddings.weight.data),
                                                                      requires_grad=True)
        self.position_embeddings_visual.weight = torch.nn.Parameter(deepcopy(self.position_embeddings.weight.data),
                                                                    requires_grad=True)
        return

    
    # My input will be:
    # question_tokens: what color is the animal here? (tokenized, i.e integer ids) (padded to the batch max sequence length)
    # image_tokens: giraffe yellow tall brown tree leafy green (tokenized, i.e. integer ids) (padded to the batch max sequence length)    
    # spatial embeddings: a spatial embedding for each of the image tokens (padded to the batch max sequence length)
    # NOTE: do I need this? visual_embeddings_type=None    
    def forward(self, question_tokens, image_tokens, spatial_embeddings, token_type_ids=None):
        """
        question_tokens = [batch_size, question_sequence_length]
        token_type_ids = [batch_size, question_sequence_length]
        image_tokens = [batch_size, image_sequence_length]
        spatial_embeddings = [batch_size, image_sequence_length, spatial_embedding_dim]
        """

        # Process question tokens as usual in BERT and similar models
        # 1. Obtain the word embeddings of the tokens
        # 2. Obtain the position embeddings of the tokens
        # 3. Build the token type embeddings
        # 4. embeddings = words_embeddings + position_embeddings + token_type_embeddings
        q_seq_length = question_tokens.size(1) 
        position_ids = torch.arange(q_seq_length, dtype=torch.long, device=question_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(question_tokens)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(question_tokens)

        q_words_embeddings = self.word_embeddings(question_tokens)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        q_embeddings = q_words_embeddings + position_embeddings + token_type_embeddings

        # Process image tokens with their spatial embeddings
        # 1. Obtain the word embeddings of the image tokens
        # 2. Project the spatial embeddings with the Linear projection layer
        # 3. Sum the word embedding and the projected spatial embedding
        #i_seq_length = image_tokens.size(1)
        i_word_embeddings = self.word_embeddings(image_tokens)
        projected_spatial_embeddings = self.projection(spatial_embeddings)
        i_embeddings = i_word_embeddings + projected_spatial_embeddings

        # Concatenate the two:
        embeddings = torch.cat((q_embeddings, i_embeddings), dim=1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSpatialModel(BertModel):
    def __init__(self, config, spatial_embedding_dim=1024):
        super(BertSpatialModel, self).__init__(config)
        self.embeddings = BertEmbeddingsWithSpatialEmbedding(config, spatial_embedding_dim=spatial_embedding_dim)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = False

        if self.bypass_transformer:
            self.additional_layer = BertLayer(config)

        self.output_attentions = config.output_attentions

        self.init_weights()

    # question_tokens, token_type_ids=None, image_tokens, spatial_embeddings
    def forward(self, question_tokens, token_type_ids, attention_mask, image_tokens, spatial_embeddings, output_all_encoded_layers=True):
        # To create the attention_mask we concatenate the question_tokens and image_tokens
        input_ids = torch.cat((question_tokens, image_tokens), axis=1)        
        if attention_mask is None:        
            attention_mask = torch.where(input_ids > 0, 1, 0).to(question_tokens.device) # NOTE: test this properly
                
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(question_tokens)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # question_tokens, image_tokens, spatial_embeddings, token_type_ids=None
        embedding_output = self.embeddings(question_tokens=question_tokens, image_tokens=image_tokens,
                                           spatial_embeddings=spatial_embeddings)
        
        if self.output_attentions:
            encoded_layers, attn_data_list = self.encoder(embedding_output,
                                                          extended_attention_mask,
                                                          output_hidden_states=output_all_encoded_layers)
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, pooled_output, attn_data_list
        else:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_hidden_states=output_all_encoded_layers)
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, pooled_output


class SpatialBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config_type, spatial_embedding_dim, num_labels, input_for_classifier):
        config = BertConfig.from_pretrained(config_type)
        super().__init__(config)

        self.num_labels = num_labels
        self.input_for_classifier = input_for_classifier # NOTE: are we putting here 'avg_pooling', 'cls'?

        self.warned = False

        self.bert = BertSpatialModel.from_pretrained(config_type, spatial_embedding_dim=spatial_embedding_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # if isinstance(config.hidden_act, str):
        #     activation_function = ACT2FN[config.hidden_act]
        # else:
        #     activation_function = config.hidden_act
        
        self.pre_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Should be done in a better way...
        model = BertModel.from_pretrained(config_type)
        self.bert.embeddings.word_embeddings = model.embeddings.word_embeddings
        self.bert.embeddings.position_embeddings = model.embeddings.position_embeddings
        self.bert.embeddings.token_type_embeddings = model.embeddings.token_type_embeddings
        
        self.bert.encoder = model.encoder
        self.bert.pooler = model.pooler

    def forward(
        self,
        question_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        image_tokens=None,
        spatial_embeddings=None        
    ):
        # question_tokens, token_type_ids, attention_mask, image_tokens, spatial_embeddings, output_all_encoded_layers=True
        outputs = self.bert(
            question_tokens,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            image_tokens=image_tokens,            
            spatial_embeddings=spatial_embeddings
        )

        if self.input_for_classifier == "cls":
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.pre_classifier(pooled_output)
            logits = self.classifier(pooled_output)
        elif self.input_for_classifier == "avg":
            sequence_output = outputs[0].last_hidden_state
            input_ids = torch.cat((question_tokens, image_tokens), axis=1)
            elem_ids = [(attention_mask[i] == 1.0).nonzero(as_tuple=True)[0] for i in range(input_ids.size(0))]
            averaged_output = torch.stack([torch.mean(sequence_output[i, elem_ids[i]], dim=0) for i in range(input_ids.size(0))])
            # averaged_output = torch.mean(outputs[0].last_hidden_state, dim=1)
            averaged_output = self.dropout(averaged_output)
            averaged_output = self.pre_classifier(averaged_output)
            logits = self.classifier(averaged_output)
        elif self.input_for_classifier == "sec":
            sequence_output = outputs[0].last_hidden_state
            if image_tokens is None:
                index_to_gather = attention_mask.sum(1) - 2
            else:
                index_to_gather = attention_mask[:,:-image_tokens.size(1)].sum(1) - 2
            pooled_output = torch.gather(
                sequence_output,
                1,
                index_to_gather.unsqueeze(-1).unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
            ).squeeze(1)
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.pre_classifier(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            if not self.warned:
                self.warned = True
                print("Warning: feeding 'cls' embedding to the classifier by default.")

            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.pre_classifier(pooled_output)
            logits = self.classifier(pooled_output)

        return logits


class BertForVQA(BertPreTrainedModel):
#class BertForVQA(nn.Module):
    def __init__(self, config_type, num_labels):
        config = BertConfig.from_pretrained(config_type)
        super().__init__(config)
        #super(BertForVQA, self).__init__()
        
        self.bert = BertModel.from_pretrained(config_type)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pre_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.pre_classifier(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
        
    
    
# Some code to test the models
from transformers import BertTokenizer

def main():
    # Test BertForVQA
    num_labels = 10
    questions = ["How many dogs are in the image?", "What color is the dog on the left?"]
    captions = ['Two dogs chasing a cat.', 'A brown dog jumping over a fence.']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    model = BertForVQA('bert-base-uncased', num_labels)

    inputs = tokenizer(questions, captions, return_tensors='pt', padding=True, truncation='only_second')
    input_ids = inputs['input_ids'].to(model.device)
    token_type_ids = inputs['token_type_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    print(f'Shape of logits: {outputs.size()}')
    sys.exit()
    
    num_labels = 10
    model = SpatialBertForSequenceClassification(
            "bert-base-uncased",
            num_labels=num_labels,
            spatial_embedding_dim=6, # If we use 'rect', we have to use 6. If 'grid', 16
            input_for_classifier='cls'
            )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



    # Simulate an input for the model
    # generate some question_tokens
    questions = ["How many dogs are in the image?", "What color is the dog on the left?"]
    question_tokens = tokenizer.batch_encode_plus(questions, add_special_tokens=True, padding=True)["input_ids"]
    question_tokens = torch.tensor(question_tokens)
    print('question tokens:')
    print(question_tokens)

    # Generate some image tokens
    obj_attr = ['dog black white dog brown standing', 'car red parked']
    image_tokens = tokenizer.batch_encode_plus(obj_attr, add_special_tokens=False, padding=True)["input_ids"]
    image_tokens = torch.tensor(image_tokens)
    print('image tokens:')
    print(image_tokens)

    # generate spatial embeddings
    #dog1 = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    #dog2 = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    #car = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    #dog1 = torch.flatten(dog1)
    #dog2 = torch.flatten(dog2)
    #car = torch.flatten(car)

    # generate spatial embeddings (rect type)
    dog1 = torch.tensor([0.5, 0.5, 0.55, 0.55, 0.05, 0.05])
    dog2 = torch.tensor([0.3, 0.3, 0.35, 0.35, 0.05, 0.05])
    car = torch.tensor([0.1, 0.8, 0.5, 0.95, 0.4, 0.15])
    dog1 = torch.flatten(dog1)
    dog2 = torch.flatten(dog2)
    car = torch.flatten(car)

    sp = [[dog1.tolist(), dog1.tolist(), dog1.tolist(), dog2.tolist(), dog2.tolist(), dog2.tolist()], [car.tolist(), car.tolist(), car.tolist(), car.tolist(), car.tolist(), car.tolist()]]
    spatial_embeddings = torch.tensor(sp, dtype=torch.float32)
    print(f'spatial embedding shape: {spatial_embeddings.size()}')

    print(f'bs: {image_tokens.size(0)} | seq_len: {image_tokens.size(1)} | spatial embedding dim: {spatial_embeddings.size(2)}')

    # Forward pass the model
    logits = model(
            question_tokens=question_tokens, # [bs, q_seq_len]            
            image_tokens=image_tokens, # [bs, i_seq_len]
            spatial_embeddings=spatial_embeddings # [bs, i_seq_len, spatial_embeddings_dim]            
            )

    print(f'logits shape: {logits.size()}')

if __name__ == "__main__":    
    main()
