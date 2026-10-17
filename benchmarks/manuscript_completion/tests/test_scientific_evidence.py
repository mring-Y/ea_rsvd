import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
audit=load('audit_factors');policy=load('matched_replay');cache=load('analyse_cached');reviewer=load('reviewer_evidence')
class ScientificTests(unittest.TestCase):
    def test_symmetric_storage_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.mtx';p.write_text('%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 2\n2 1 3\n2 1 1\n')
            m,n,o,c,v=audit.read_mm(p)
            self.assertEqual(o.tolist(),[0,2,3]);self.assertEqual(c.tolist(),[0,1,0]);self.assertEqual(v.tolist(),[2,4,4])
    def test_projection_pass_cannot_accept_failed_output_rank(self):
        states={(w,q):dict(width=w,power=q,rho_l=.1,rho_k=.5-q*.1,cost_ms=1,valid=True) for w in (24,32,40) for q in (0,1,2)}
        result,steps=policy.replay(states,'rank_residual');self.assertFalse(result['attained']);self.assertEqual(result['evaluated_states'],9);self.assertAlmostEqual(result['output_rho_k'],.3)
        result,steps=policy.replay(states,'residual_pair');self.assertFalse(result['attained']);self.assertEqual((steps[1]['width'],steps[1]['power']),(32,0))
    def test_nonorthogonal_factor_gram_identity_matches_reconstruction(self):
        o=np.array([0,1,2]);c=np.array([0,1]);v=np.array([3.,1.]);u=np.array([[1.01],[.02]]);s=np.array([2.9]);right=np.array([[.99],[.01]])
        self.assertAlmostEqual(audit.direct_residual(2,2,o,c,v,u,s,right),audit.gram_residual(2,2,o,c,v,u,s,right),places=13)
    def test_failed_candidate_is_not_revisited(self):
        states={(w,q):dict(width=w,power=q,rho_l=.1,rho_k=.5,cost_ms=1,valid=True) for w in (24,32,40) for q in (0,1,2)}
        states[(32,0)]['valid']=False
        result,steps=policy.replay(states,'residual_pair')
        visited=[(r['width'],r['power']) for r in steps]
        self.assertEqual(len(visited),len(set(visited)))
    def test_unknown_failure_cost_is_not_zero(self):
        states={(w,q):dict(width=w,power=q,rho_l=.8,rho_k=.9,cost_ms=1,valid=True) for w in (24,32,40) for q in (0,1,2)}
        states[(24,0)].update(valid=False,cost_ms=float('nan'))
        result,steps=policy.replay(states,'rank_residual')
        self.assertIsNone(result['cumulative_ms'])
    def test_saved_cache_samples_rebuild_all_reported_medians(self):
        paths=list((ROOT/'evidence/cache').rglob('*.csv'));paths=[p for p in paths if not p.name.endswith('.samples.csv')]
        self.assertEqual(len(paths),18)
        for p in paths:cache.validate_samples(p)
    def test_current_subspace_rank_label_does_not_prove_global_rank_failure(self):
        a=np.diag([1.,.18,.01]);angle=.15
        q=np.array([[np.cos(angle),0.],[0.,1.],[np.sin(angle),0.]])
        b=q.T@a;bu,s,bv=np.linalg.svd(b,full_matrices=False)
        norm=np.linalg.norm(a)
        rho_l=np.linalg.norm(a-q@b)/norm
        rho_k=np.linalg.norm(a-q@bu[:,:1]@np.diag(s[:1])@bv[:1,:])/norm
        optimal=np.linalg.norm(np.linalg.svd(a,compute_uv=False)[1:])/norm
        self.assertLess(rho_l,.20);self.assertGreater(rho_k,.20);self.assertLess(optimal,.20)
        self.assertLessEqual(reviewer.tail_lower_bound(rho_k,rho_l),optimal+1e-12)
    def test_projected_tail_lower_bound_matches_independent_optimal_svd(self):
        rng=np.random.default_rng(7241)
        for _ in range(6):
            a=rng.normal(size=(10,7));q=np.linalg.qr(rng.normal(size=(10,4)))[0]
            b=q.T@a;bu,s,bv=np.linalg.svd(b,full_matrices=False);norm=np.linalg.norm(a)
            rho_l=np.linalg.norm(a-q@b)/norm
            rho_k=np.linalg.norm(a-q@bu[:,:2]@np.diag(s[:2])@bv[:2,:])/norm
            optimal=np.linalg.norm(np.linalg.svd(a,compute_uv=False)[2:])/norm
            lower=reviewer.tail_lower_bound(rho_k,rho_l)
            self.assertLessEqual(lower,optimal+1e-12);self.assertLessEqual(optimal,rho_k+1e-12)
if __name__=='__main__':unittest.main()
