from __future__ import unicode_literals, print_function
from django.shortcuts import render
from django.views.generic import View
from django.http import HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from retrain_fer.models import QueryDB, EntityDB
# training

import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm # loading bar

#custom
from retrain_fer.mixin import SerializerMixin, HttpresponseMixin  
from retrain_fer.forms import QueryDBForm, EntityDBForm

import pdb

#======================================================================================================
# REST API
#______________________________________________________________________________________________________
# Save Training data to db
# ========================
# table 1: QueryDB
# attributes: id, query
# table 2: EntityDB
# attributes: id, query(foregn field:- QueryDB),entity_name, start_pos, end_pos
# input format
# ============
# {
#   'query' : <query :: str>,
#   'entity_list' : [(<start pos :: int>, <end pos :: int>, <entity :: str>)]
# }
# eg:
# {
#   'query' :"Horses are too tall and horses are pretend to care about your feelings",
#   'entity_list' : [(0, 6, 'ANIMAL'), (25, 31, 'ANIMAL')]
# }
# output
# ======
# {
#   'result' : result,
#   'error' : error
# }
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class FerSaveData2DB(HttpresponseMixin, SerializerMixin, View):
    def get_object_by_query(self,query):
        try:
            emp = QueryDB.objects.get(query=query)
        except QueryDB.DoesNotExist:
            emp = None
        return emp
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs):
        result = []
        error = []   
        flag = True
        
        if 'query' in request.POST:
            query = request.POST['query']
        else:
            error.append('query is not given')
            flag = False
            status = 400
        if 'entity_list' in request.POST:
            entity_list = request.POST.getlist('entity_list')
        else:
            error.append('entity is not given')
            flag = False
            status = 400
       
        if flag:            
            query_data = {                
                'query' : query
            } 
            query_form = QueryDBForm(query_data)
            # pdb.set_trace()
            if query_form.is_valid():
                query_form.save(commit=True)                
                
            if query_form.errors:
                error.append(query_form.errors)
                status = 400
            # save entity
            
            for i in range(0,len(entity_list),3):
                entity_data = {                    
                    'query' : query,
                    'entity_name' : entity_list[i+2],
                    'start_pos' : int(entity_list[i+0]),
                    'end_pos' : int(entity_list[i+1])
                } 
                entity_form = EntityDBForm(entity_data)
                if entity_form.is_valid():
                    entity_form.save(commit=True)
                    result.append('%s(%s,%s) added'%(entity_list[i+2], entity_list[i+0], entity_list[i+1]))                      
                    status = 200
                if entity_form.errors:
                    error.append(entity_form.errors)
                    status = 400
        f_result = {
            'result' : result,
            'error' : error
        }   
        json_data = json.dumps(f_result)
        return self.render_to_http_response(json_data, status=status)
#______________________________________________________________________________________________________
# new entity label
LABEL = 'ANIMAL'
TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("Do they bite?", {
        'entities': []
    }),

    ("horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("horses pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("they pretend to care about your feelings, those horses", {
        'entities': [(48, 54, 'ANIMAL')]
    }),

    ("horses?", {
        'entities': [(0, 6, 'ANIMAL')]
    })
]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def train(TRAIN_DATA, model=None, new_model_name='animal', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm(TRAIN_DATA):
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model
    test_text = 'Do you like horses?'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)
#______________________________________________________________________________________________________
@method_decorator(csrf_exempt, name='dispatch')
class TrainData(HttpresponseMixin, SerializerMixin, View):
    def get(self, request, *args, **kwargs):
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
    def post(self, request, *args, **kwargs):
        # TRAIN_DATA creation
        train()
        json_data = json.dumps({'msg':'This is from get method'})
        return HttpResponse(json_data, content_type='application/json')
#______________________________________________________________________________________________________