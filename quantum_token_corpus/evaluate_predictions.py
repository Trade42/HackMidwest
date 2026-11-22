#!/usr/bin/env python3
"""Evaluate id,prediction CSV against a labeled corpus; Python standard library only."""
import argparse
import csv
from collections import defaultdict

p=argparse.ArgumentParser()
p.add_argument('predictions',help='CSV with id,prediction columns (0 or 1)')
p.add_argument('--gold',default='test_mixed.csv',help='Labeled test CSV')
a=p.parse_args()
with open(a.gold, newline='', encoding='utf-8') as f:
    gold={r['id']:r for r in csv.DictReader(f)}
with open(a.predictions,newline='',encoding='utf-8') as f:
    predrows=list(csv.DictReader(f))
pred={}
for r in predrows:
    if r['id'] in pred:
        raise ValueError('Duplicate prediction ID: '+r['id'])
    pred[r['id']]=int(r['prediction'])
    if pred[r['id']] not in (0,1): raise ValueError('Prediction must be 0 or 1')
extra=set(pred)-set(gold)
missing=set(gold)-set(pred)
if extra or missing:
    raise ValueError(f'{len(extra)} extra and {len(missing)} missing prediction IDs')
counts=defaultdict(lambda:[0,0])
for id,r in gold.items():
    correct=int(pred[id]==int(r['label']))
    for k in ('overall',f"tier_{r['tier']}"):
        counts[k][0]+=correct
        counts[k][1]+=1
for key in ['overall','tier_1','tier_2','tier_3']:
    good,total=counts[key]
    print(f'{key}: {good}/{total} = {good/total:.2%}')
