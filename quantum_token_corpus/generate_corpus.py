#!/usr/bin/env python3
"""Reproducible controlled-coreference dataset for quantum/classical token interaction research.
No dependencies beyond Python standard library. Run: python generate_corpus.py
"""
import csv
import json
import random
from collections import Counter
from itertools import product
from pathlib import Path

SEED = 241121
RNG = random.Random(SEED)
OUT = Path(__file__).resolve().parent
COLORS = [('red','blue'), ('green','yellow'), ('silver','gold')]

T1_PAIRS = [
 ('trophy','suitcase'), ('parcel','locker'), ('foam insert','compartment'),
 ('cushion','storage box'), ('toolkit','drawer'), ('package','crate'),
 ('sample vial','holder'), ('notebook','pouch'), ('lunchbox','backpack'),
 ('bottle','cooler'), ('painting','frame'), ('wooden block','mold'),
 ('tablet','sleeve'), ('phone','case'), ('laptop','bag'),
 ('file folder','cabinet'), ('toy','container'), ('soap bar','tin'),
 ('book','box'), ('camera','carrying case'), ('keychain','pocket'),
 ('canister','cubby'), ('sculpture','display case')]
T2_PAIRS = [
 ('rug','floor'), ('blanket','bed'), ('tablecloth','table'), ('poster','noticeboard'),
 ('screen protector','display'), ('protective film','window'),
 ('canvas sheet','frame'), ('cover sheet','panel'), ('tarp','platform'),
 ('fabric sheet','workbench'), ('carpet','landing'), ('awning','patio'),
 ('shade cloth','deck'), ('sticker','label area'), ('wrapping sheet','package'),
 ('roof membrane','roof'), ('mat','shelf'), ('plastic liner','tray'),
 ('insulation sheet','wall'), ('floor tile','floor section'),
 ('cloth cover','bench'), ('paper overlay','diagram'), ('vinyl wrap','door')]
# Semantic noun domains whose underlying settings can plausibly be exchanged.
T3_DOMAINS = [
 ('lamp','brightness settings','brighter','dimmer'),
 ('screen','brightness settings','brighter','dimmer'),
 ('signal light','brightness settings','brighter','dimmer'),
 ('monitor','brightness settings','brighter','dimmer'),
 ('beaker','temperatures','warmer','cooler'),
 ('tank','temperatures','warmer','cooler'),
 ('flask','temperatures','warmer','cooler'),
 ('cup','temperatures','warmer','cooler'),
 ('chamber','temperatures','warmer','cooler'),
 ('tank','pressure settings','higher-pressure','lower-pressure'),
 ('cylinder','pressure settings','higher-pressure','lower-pressure'),
 ('pipe','pressure settings','higher-pressure','lower-pressure'),
 ('chamber','pressure settings','higher-pressure','lower-pressure'),
 ('fan','speed settings','faster','slower'),
 ('motor','speed settings','faster','slower'),
 ('conveyor','speed settings','faster','slower'),
 ('wheel','speed settings','faster','slower'),
 ('speaker','volume settings','louder','quieter'),
 ('alarm','volume settings','louder','quieter'),
 ('buzzer','volume settings','louder','quieter'),
 ('battery','voltage settings','higher-voltage','lower-voltage'),
 ('cell','voltage settings','higher-voltage','lower-voltage'),
 ('device','voltage settings','higher-voltage','lower-voltage')]
assert len(T1_PAIRS)==len(T2_PAIRS)==len(T3_DOMAINS)==23

# 45 train scenarios, 9 validation, 9 test, 6 paraphrase/OOD test,
# with exactly 8 balanced examples per scenario (per tier).
# Each base noun combination occurs in training, and scenarios (which include
# descriptor colors) never cross splits.
scenario_assignment = {}
base_indices = list(range(23))
RNG.shuffle(base_indices)
for rank, base_index in enumerate(base_indices):
    scenario_assignment[(base_index,0)] = 'train'
    scenario_assignment[(base_index,1)] = 'train' if rank < 22 else 'test_ood'
    if rank < 9:
        scenario_assignment[(base_index,2)] = 'validation'
    elif rank < 18:
        scenario_assignment[(base_index,2)] = 'test'
    else:
        scenario_assignment[(base_index,2)] = 'test_ood'
assert Counter(scenario_assignment.values()) == dict(train=45,validation=9,test=9,test_ood=6)

# Name-level colors shuffle per base to ensure each base pair's color variants
# have unrelated split assignments (but all labels remain perfectly balanced).
color_assignments = {}
for base_idx in range(23):
    color_idx = list(range(3))
    RNG.shuffle(color_idx)
    color_assignments[base_idx] = color_idx


def t1_text(a,b,adj,style):
    templates = [
        'The {a} did not fit inside the {b} because it was too {adj}.',
        'The {a} could not fit into the {b} because it was too {adj}.',
        'The {a} would not fit inside the {b} since it was too {adj}.',
        'The {a} was unable to fit into the {b} because it was too {adj}.',
        'The {a} could not be placed inside the {b}, since it was too {adj}.', # OOD
        'The {a} would not go into the {b}, as it was too {adj}.', # OOD
        'The {a} could not be stored in the {b} because it was too {adj}.', # OOD
        'The {a} remained outside the {b} since it was too {adj}.' # OOD
    ]
    return templates[style].format(a=a,b=b,adj=adj)


def t2_text(a,b,relation,adj,style):
    if style == 0:
        phrase = 'could not fit on' if relation=='fit' else 'could not cover'
        return f'The {a} {phrase} the {b} because it was too {adj}.'
    if style == 1:
        phrase = 'failed to fit on' if relation=='fit' else 'failed to cover'
        return f'The {a} {phrase} the {b} since it was too {adj}.'
    # Out-of-distribution wording; semantic rule is identical.
    phrase = 'was unable to fit onto' if relation=='fit' else 'was unable to cover'
    return f'The {a} {phrase} the {b}, as it was too {adj}.'


def property_phrases(prop, high, low):
    if high in ('higher-pressure','higher-voltage'):
        noun = 'pressure' if 'pressure' in high else 'voltage'
        return (f'under higher {noun}' if noun=='pressure' else f'at higher {noun}',
                f'under lower {noun}' if noun=='pressure' else f'at lower {noun}',
                f'under higher {noun}' if noun=='pressure' else f'at higher {noun}',
                f'under lower {noun}' if noun=='pressure' else f'at lower {noun}',
                False)
    return (high,low,high,low,True)


def t3_text(a,b,property_name,high,low,initial_high,swap,query_high,style):
    initial_high_phrase, initial_low_phrase, query_high_phrase, query_low_phrase, ordinary_adj = property_phrases(property_name,high,low)
    initial = initial_high_phrase if initial_high else initial_low_phrase
    query = query_high_phrase if query_high else query_low_phrase
    changed = 'were exchanged' if swap else 'were left unchanged'
    changed_active = f'exchanged their {property_name}' if swap else f'left their {property_name} unchanged'
    if style == 0:
        start = f'Initially, the {a} was {initial} than the {b}.'
        end = (f'Afterward, the researcher inspected it because it was the {query} one.'
               if ordinary_adj else
               f'Afterward, the researcher inspected it because it was {query}.')
        return f'{start} During calibration, their {property_name} {changed}. {end}'
    if style == 1:
        start = f'Before the test, the {a} was {initial} than the {b}.'
        end = (f'In the final check, the engineer selected it because it was the {query} one.'
               if ordinary_adj else
               f'In the final check, the engineer selected it because it was {query}.')
        return f'{start} The team then {changed_active}. {end}'
    if style == 2:
        start = f'At first, the {a} was {initial} than the {b}.'
        end = (f'Later, the technician examined it because it was the {query} one.'
               if ordinary_adj else
               f'Later, the technician examined it because it was {query}.')
        return f'{start} During adjustment, their {property_name} {changed}. {end}'
    # Paraphrase OOD, not seen in training.
    start = f'To begin with, the {a} was {initial} than the {b}.'
    end = (f'Following the change, the scientist chose it for inspection because it was the {query} one.'
           if ordinary_adj else
           f'Following the change, the scientist chose it for inspection because it was {query}.')
    return f'{start} The technicians then {changed_active}. {end}'


def new_record(tier,split,scenario,candidate_0,candidate_1,sentence,label,template, factors):
    return dict(
        id=f'T{tier}-{scenario}-{len(records)+1:05d}',
        tier=tier, split=split, sentence=sentence,
        candidate_0=candidate_0, candidate_1=candidate_1,
        label=label, referent=(candidate_0 if label==0 else candidate_1),
        scenario_id=scenario, template_id=template,
        factors=json.dumps(factors,sort_keys=True,separators=(',',':'))
    )

records=[]
for tier in (1,2,3):
    for base_idx in range(23):
        for variant_idx in range(3):
            split=scenario_assignment[(base_idx,variant_idx)]
            colors=COLORS[color_assignments[base_idx][variant_idx]]
            scenario=f'T{tier}-B{base_idx:02d}-C{variant_idx}'
            if tier==1:
                base_a,base_b=T1_PAIRS[base_idx]
            elif tier==2:
                base_a,base_b=T2_PAIRS[base_idx]
            else:
                base_a,prop,hi,lo=T3_DOMAINS[base_idx]
                base_b=base_a
            a=f'{colors[0]} {base_a}'
            b=f'{colors[1]} {base_b}'
            if tier==1:
                for adjective in ('large','small'):
                    for template in range(4):
                        style=4+template if split=='test_ood' else template
                        sentence=t1_text(a,b,adjective,style)
                        label=0 if adjective=='large' else 1
                        factors={'size_cue':adjective,'rule':'oversized_item_or_undersized_container'}
                        records.append(new_record(tier,split,scenario,a,b,sentence,label,f't1-{style}',factors))
            elif tier==2:
                for relation,adjective,templ in product(('fit','cover'),('large','small'),range(2)):
                    style=2 if split=='test_ood' else templ
                    # Two copies are needed for each semantic assignment; in the OOD split
                    # use different causal-link wording, not duplicate surface sentences.
                    if split=='test_ood' and templ==1:
                        sentence=t2_text(a,b,relation,adjective,2).replace(', as it', ', given that it')
                        style='2b'
                    else:
                        sentence=t2_text(a,b,relation,adjective,style)
                    label=0 if ((relation=='fit')==(adjective=='large')) else 1
                    factors={'relation':relation,'size_cue':adjective,'rule':'fit_cover_relation_adjective_interaction'}
                    records.append(new_record(tier,split,scenario,a,b,sentence,label,f't2-{style}',factors))
            else:
                style=3 if split=='test_ood' else (base_idx+variant_idx)%3
                for initial_high,swap,query_high in product((False,True),repeat=3):
                    sentence=t3_text(a,b,prop,hi,lo,initial_high,swap,query_high,style)
                    final_left_high=initial_high ^ swap
                    label=0 if final_left_high==query_high else 1
                    factors={'initial_left_high':int(initial_high),'swapped':int(swap),'query_high':int(query_high),'property':prop,'rule':'initial_comparison_x_swap_x_query'}
                    records.append(new_record(tier,split,scenario,a,b,sentence,label,f't3-{style}',factors))

# Prevent leakage: different splits never share the same scenario, and labels balance
# at every tier and split. Every sentence unique globally.
assert len(records)==3*69*8
assert len({r['sentence'] for r in records})==len(records), 'Duplicate generated sentences'
assert len({r['id'] for r in records})==len(records)
for tier in (1,2,3):
    for split,n in [('train',360),('validation',72),('test',72),('test_ood',48)]:
        rs=[r for r in records if r['tier']==tier and r['split']==split]
        assert len(rs)==n,(tier,split,len(rs))
        assert Counter(r['label'] for r in rs)=={0:n//2,1:n//2}
        assert len({r['scenario_id'] for r in rs})==n//8
for tier in (1,2,3):
    for s1 in ('train','validation','test','test_ood'):
        for s2 in ('train','validation','test','test_ood'):
            if s1 < s2:
                ids1={r['scenario_id'] for r in records if r['tier']==tier and r['split']==s1}
                ids2={r['scenario_id'] for r in records if r['tier']==tier and r['split']==s2}
                assert ids1.isdisjoint(ids2)

FIELDS=['id','tier','sentence','candidate_0','candidate_1','label','referent','scenario_id']
META_FIELDS=['id','tier','split','scenario_id','template_id','factors']

def write_csv(name,data,fields=FIELDS):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(data)

def write_jsonl(name,data):
    with (OUT/name).open('w',encoding='utf-8') as f:
        for r in data:
            obj={k:r[k] for k in FIELDS}
            obj['label_text']=r['referent']
            f.write(json.dumps(obj,ensure_ascii=False)+'\n')

# Shuffle within files, reproducibly.
for split in ('train','validation','test','test_ood'):
    data=[r for r in records if r['split']==split]
    RNG.shuffle(data)
    write_csv(f'{split}_mixed.csv',data)
    if split in ('test', 'test_ood'):
        write_csv(f'{split}_mixed_unlabeled.csv', data, ['id','tier','sentence','candidate_0','candidate_1'])
    write_jsonl(f'{split}_mixed.jsonl',data)
    for tier in (1,2,3):
        sub=[r for r in data if r['tier']==tier]
        write_csv(f'tier_{tier}_{split}.csv',sub)

# A small resource-conscious starter dataset for first quantum simulations.
for split,size_each in [('train',48),('validation',24),('test',24)]:
    sub=[]
    for tier in (1,2,3):
        pool=[r for r in records if r['tier']==tier and r['split']==split]
        RNG.shuffle(pool)
        # Sample entire scenario blocks (8 examples) to preserve all truth-table
        # combinations in every split.
        scenarios=list(dict.fromkeys(r['scenario_id'] for r in pool))
        take=set(scenarios[:size_each//8])
        tier_sub=[r for r in pool if r['scenario_id'] in take]
        assert len(tier_sub)==size_each
        assert Counter(r['label'] for r in tier_sub)=={0:size_each//2,1:size_each//2}
        sub.extend(tier_sub)
    RNG.shuffle(sub)
    write_csv(f'{split}_mixed_small.csv',sub)

write_csv('metadata_all.csv',records,META_FIELDS)
summary={
    'seed':SEED,
    'primary_input_columns':['sentence','candidate_0','candidate_1'],
    'target_column':'label',
    'label_mapping':'0 = candidate_0; 1 = candidate_1',
    'counts':{split:{str(tier):sum(r['split']==split and r['tier']==tier for r in records) for tier in (1,2,3)} for split in ('train','validation','test','test_ood')},
    'notes':'Only input columns are model inputs. Never train on referent, scenario_id, tier, or metadata_all.csv. Training and eval scenario IDs do not overlap.'
}
(OUT/'dataset_manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
print('Examples by tier:')
for tier in (1,2,3):
    for r in (r for r in records if r['tier']==tier and r['split']=='train'):
        print(f"  Tier {tier}: {r['sentence']} -> {r['referent']}")
        break
