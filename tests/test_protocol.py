import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from solubility_public_data.models import FusionModel, masked_pool
from solubility_public_data.data import FeatureStore, subset
from solubility_public_data.training import Estimator, inner_split
from solubility_public_data.evaluation import metrics, choose_threshold, make_splits, validate_splits


@pytest.fixture
def fixture(tmp_path):
    rng=np.random.default_rng(1);n=60;lengths=np.full(n,20);ids=np.array([f'test_{i}' for i in range(n)])
    seq=np.array(['ACDEFGHIKLMNPQRSTVW'+('A' if i%2 else 'C') for i in range(n)])
    np.savez(tmp_path/'pooled_20261108.npz',ids=ids,sequences=seq,desc=rng.normal(size=(n,202)).astype('f4'),esm650=rng.normal(size=(n,1280)).astype('f4'),esm650_max=rng.normal(size=(n,1280)).astype('f4'),offsets=np.r_[0,np.cumsum(lengths)])
    np.save(tmp_path/'residues_20261108.npy',rng.normal(size=(n*20,1280)).astype('f2'))
    # The fixture is synthetic and excluded from scientific metrics.
    s=dict(ids=ids,sequence_sha256=np.array([hashlib.sha256(s.encode()).hexdigest() for s in seq]),global_values=rng.normal(size=(n,4)),global_available=np.ones(n),residue_values=rng.normal(size=(n*20,3)),residue_offsets=np.arange(n+1)*20,residue_mask=np.ones(n*20,bool),residue_available=np.ones(n),patch_values=rng.normal(size=(n*2,5)),patch_offsets=np.arange(n+1)*2,patch_mask=np.ones(n*2,bool),patch_available=np.ones(n),quality=np.ones((n,3)),global_feature_names=np.array(['g0','g1','g2','g3']),residue_feature_names=np.array(['r0','r1','r2']),patch_feature_names=np.array(['p0','p1','p2','p3','p4']))
    np.savez(tmp_path/'structure.npz',**s)
    records=[dict(record_id=str(i),sequence=str(s),insoluble=j%2,group_id=f'group_{j}',role='train') for j,(i,s) in enumerate(zip(ids,seq))]
    return tmp_path,records


@pytest.mark.parametrize('architecture,expected',[('residual_mlp',244387),('bilstm',393284)])
def test_public_parameter_counts(architecture,expected):
    model=FusionModel(architecture)
    assert sum(p.numel() for p in model.parameters())==expected


@pytest.mark.parametrize('architecture',['residual_mlp','bilstm'])
@pytest.mark.parametrize('profile',['public','structure'])
def test_fit_reload_and_training_only_scalers(fixture,architecture,profile):
    root,records=fixture
    store=FeatureStore(root,root/'structure.npz' if profile=='structure' else None)
    data=store.align(records,architecture=='bilstm',profile=='structure')
    y=np.array([r['insoluble'] for r in records],np.float32);groups=np.array([r['group_id'] for r in records])
    config=dict(architecture=architecture,profile=profile,max_epochs=1,patience=1,batch_size=32)
    tr=np.arange(48);model=Estimator(config,store).fit(subset(data,tr),y[tr],groups[tr]);p=model.predict(data)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    assert model.metadata['inner_group_overlap']==0
    assert np.allclose(model.scalers['esm650']['mean'],data['esm650'][tr].mean(0),atol=1e-6)
    model.save(root/'model.pt',.4);loaded=Estimator.load(root/'model.pt',store)
    assert np.array_equal(p,loaded.predict(data)) and loaded.threshold==.4
    # A model checkpoint accepts a new feature store: it is not tied to training IDs.
    if profile == 'public':
        import os
        other=root/'new_candidates';other.mkdir()
        changed={k:v.copy() for k,v in store.pooled.items()}
        changed['ids']=np.array(['candidate_'+str(i) for i in range(len(records))])
        np.savez(other/'pooled_20261108.npz',**changed)
        os.link(root/'residues_20261108.npy',other/'residues_20261108.npy')
        new_records=[{**r,'record_id':str(i)} for r,i in zip(records,changed['ids'])]
        new_store=FeatureStore(other);new_model=Estimator.load(root/'model.pt',new_store)
        assert np.array_equal(p,new_model.predict(new_store.align(new_records,architecture=='bilstm')))
    assert model.metadata['n_training']==48


def test_bilstm_padding_invariance():
    torch.manual_seed(1);model=FusionModel('bilstm').eval()
    b={'desc':torch.randn(2,202),'esm650':torch.randn(2,1280),'esm650_max':torch.randn(2,1280),'residue':torch.randn(2,25,1280),'lengths':torch.tensor([20,25]),'mask':torch.arange(25)[None,:]<torch.tensor([20,25])[:,None]}
    with torch.no_grad():
        a=model(b)['logit'];b['residue']=torch.nn.functional.pad(b['residue'],(0,0,0,13),value=float('nan'));b['mask']=torch.nn.functional.pad(b['mask'],(0,13),value=False)
        assert torch.allclose(a,model(b)['logit'],atol=1e-7)


def test_patch_permutation_and_missing_structure(fixture):
    root,records=fixture;store=FeatureStore(root,root/'structure.npz');data=store.align(records,False,True)
    b=store.batch(data,np.arange(4),store.scalers(data,np.arange(48)))
    model=FusionModel(profile='structure',structure_dims=store.structure_dims).eval()
    with torch.no_grad():
        a=model(b)['logit'];b['structure_patch']=b['structure_patch'].flip(1);b['structure_patch_mask']=b['structure_patch_mask'].flip(1)
        assert torch.equal(a,model(b)['logit'])
        b['structure_available'].zero_();b['structure_residue_mask'].zero_();b['structure_patch_mask'].zero_();b['structure_quality'].zero_()
        result=model(b);assert torch.isfinite(result['logit']).all();assert torch.all(result['gates'][:,0]==0)


def test_threshold_polarity_and_empty_acceptance():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.7,.9]);t=choose_threshold(y,p,1)
    m=metrics(y,p,t);assert t==.7 and m['recall']==1 and m['soluble_retention']==1 and m['accepted_insoluble_fraction']==0
    assert metrics(y,p,0)['accepted_insoluble_fraction'] is None


def test_five_distinct_holdouts(fixture):
    _,records=fixture
    for mode in ['random','grouped']:
        s=make_splits(records,mode);assert len({tuple(sorted(v['test'])) for v in s.values()})==5


def test_id_sequence_guard(fixture):
    root,records=fixture;store=FeatureStore(root);records[0]['sequence']='A'*20
    with pytest.raises(ValueError,match='Sequence mismatch'):store.align(records)


def test_group_leakage_is_rejected(fixture):
    _,records=fixture;s=make_splits(records,'grouped')
    tr,te=s['11']['train'],s['11']['test']
    records[tr[0]]['group_id']=records[te[0]]['group_id']
    with pytest.raises(ValueError,match='Family leakage'):validate_splits(records,s,'grouped')


def test_inner_selection_does_not_receive_holdout_labels():
    y=np.array([0,1]*30);g=np.arange(60).astype(str)
    tr,va=inner_split(y[:48],g[:48],11);y[48:]=1-y[48:]
    tr2,va2=inner_split(y[:48],g[:48],11)
    assert np.array_equal(tr,tr2) and np.array_equal(va,va2)
