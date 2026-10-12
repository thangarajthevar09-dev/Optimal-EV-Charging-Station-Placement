"""EV charging station placement decision-support prototype.

Uses illustrative relative coordinates and synthetic demand weights. Results are
for coursework demonstration only and are not validated geographic coverage.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
LEVEL = {"Low": 1, "Medium": 2, "High": 3}
RADIUS_KM = 2.0

def load_inputs(candidate_path, zones_path):
    c = pd.read_csv(candidate_path)
    z = pd.read_csv(zones_path)
    required_c = {"ID", "Candidate area", "Traffic", "Existing coverage", "Travel demand", "Grid capacity", "Illustrative cost units", "X_km_illustrative", "Y_km_illustrative"}
    required_z = {"Zone_ID", "Demand zone", "X_km_illustrative", "Y_km_illustrative", "Demand_weight_illustrative", "Demand_level"}
    if not required_c.issubset(c.columns): raise ValueError(f"Candidate CSV missing: {sorted(required_c-set(c.columns))}")
    if not required_z.issubset(z.columns): raise ValueError(f"Demand-zone CSV missing: {sorted(required_z-set(z.columns))}")
    for col in ["Traffic", "Existing coverage", "Travel demand", "Grid capacity"]:
        if not c[col].isin(LEVEL).all(): raise ValueError(f"Unexpected category in {col}; use Low/Medium/High")
    if (pd.to_numeric(c['Illustrative cost units']) <= 0).any(): raise ValueError('Costs must be positive')
    if (z['Demand_weight_illustrative'] < 0).any(): raise ValueError('Demand weights must be nonnegative')
    return c, z

def distance_matrix(c, z):
    cx=c['X_km_illustrative'].to_numpy()[:,None]; cy=c['Y_km_illustrative'].to_numpy()[:,None]
    zx=z['X_km_illustrative'].to_numpy()[None,:]; zy=z['Y_km_illustrative'].to_numpy()[None,:]
    return np.sqrt((cx-zx)**2+(cy-zy)**2)

def score_candidates(c):
    r=c.copy()
    r['Traffic_score']=r['Traffic'].map(LEVEL)
    r['Existing_coverage_score']=r['Existing coverage'].map(LEVEL)
    r['Travel_demand_score']=r['Travel demand'].map(LEVEL)
    r['Grid_capacity_score']=r['Grid capacity'].map(LEVEL)
    # Higher traffic/demand, low existing coverage and stronger grid capacity improve score.
    r['Priority_score']=(.20*r['Traffic_score']/3 + .35*r['Travel_demand_score']/3 + .30*(4-r['Existing_coverage_score'])/3 + .15*r['Grid_capacity_score']/3)*100
    r['Grid_feasible']=r['Grid capacity'].ne('Low')
    r['Score_per_cost']=r['Priority_score']/r['Illustrative cost units']
    return r.sort_values(['Grid_feasible','Score_per_cost','Priority_score'],ascending=False).reset_index(drop=True)

def greedy_select(candidates, ranked, zones, distances, budget, radius=RADIUS_KM):
    feasible=ranked[ranked['Grid_feasible']].copy()
    weights=zones['Demand_weight_illustrative'].to_numpy(float)
    covered=np.zeros(len(zones),dtype=bool); chosen=[]; spent=0.0
    # Greedy maximum marginal uncovered demand benefit per cost, with score as tie-breaker.
    remaining=set(feasible['ID'])
    while remaining:
        options=[]
        for cid in remaining:
            row=feasible.loc[feasible.ID.eq(cid)].iloc[0]
            i=int(candidates.index[candidates.ID.eq(cid)][0])
            newly=(distances[i] <= radius) & (~covered)
            marginal=float(weights[newly].sum())
            # small, transparent priority contribution breaks ties without dominating demand coverage
            benefit=marginal + float(row['Priority_score'])*0.20
            cost=float(row['Illustrative cost units'])
            options.append((benefit/cost, marginal, float(row['Priority_score']), cid, cost, newly))
        options.sort(key=lambda x:(x[0],x[1],x[2]),reverse=True)
        picked=False
        for ratio,marginal,score,cid,cost,newly in options:
            if spent+cost <= budget:
                chosen.append(cid); spent+=cost; covered |= (distances[int(candidates.index[candidates.ID.eq(cid)][0])] <= radius)
                remaining.remove(cid); picked=True; break
            remaining.remove(cid)
        if not picked and not remaining: break
    selected=ranked[ranked.ID.isin(chosen)].copy()
    # keep selection order by greedy sequence
    selected['Selection_order']=selected.ID.map({cid:i+1 for i,cid in enumerate(chosen)})
    return selected.sort_values('Selection_order'), covered

def make_outputs(c,z,budgets,radius):
    OUT.mkdir(parents=True,exist_ok=True)
    ranked=score_candidates(c)
    dists=distance_matrix(c,z)
    ranked.to_csv(OUT/'candidate_ranking.csv',index=False)
    # Per candidate zone coverage, useful for auditability.
    coverage_rows=[]
    for i,row in c.iterrows():
        for j,zr in z.iterrows():
            dist=float(dists[i,j])
            coverage_rows.append({'Candidate_ID':row.ID,'Candidate_area':row['Candidate area'],'Zone_ID':zr.Zone_ID,'Demand_zone':zr['Demand zone'],'Distance_km_illustrative':round(dist,3),'Within_radius':bool(dist<=radius),'Demand_weight_illustrative':zr.Demand_weight_illustrative})
    pd.DataFrame(coverage_rows).to_csv(OUT/'candidate_zone_distances.csv',index=False)
    summary=[]; scenario_cover={}
    labels=['Low','Medium','High'] if len(budgets)==3 else [f'Scenario_{i+1}' for i in range(len(budgets))]
    for label,budget in zip(labels,budgets):
        selected,covered=greedy_select(c,ranked,z,dists,budget,radius)
        selected.to_csv(OUT/f'selected_locations_{label.lower()}.csv',index=False)
        scenario_cover[label]=covered
        total_weight=float(z['Demand_weight_illustrative'].sum())
        covered_weight=float(z.loc[covered,'Demand_weight_illustrative'].sum())
        # utilization is a transparent proxy: selected-site average demand score, not measured usage.
        demand_score=selected['Travel demand'].map(LEVEL).mean() if len(selected) else 0
        util_proxy=round(35+(demand_score/3)*50,1) if len(selected) else 0.0
        cost=float(selected['Illustrative cost units'].sum()) if len(selected) else 0.0
        summary.append({'Budget_scenario':label,'Budget_units_illustrative':budget,'Stations_selected':len(selected),'Cost_used_units':round(cost,2),'Budget_remaining_units':round(budget-cost,2),'Demand_zones_covered':int(covered.sum()),'Demand_zones_total':len(z),'Demand_weight_covered':round(covered_weight,1),'Demand_weight_total':round(total_weight,1),'Weighted_coverage_percent_proxy':round(100*covered_weight/total_weight,1) if total_weight else 0,'Zone_count_coverage_percent_proxy':round(100*covered.sum()/len(z),1) if len(z) else 0,'Utilization_percent_proxy':util_proxy,'Selected_areas':'; '.join(selected['Candidate area'].tolist()) if len(selected) else 'None'})
        cov=z.copy(); cov['Covered']=covered; cov['Nearest_selected_distance_km_illustrative']=[min([dists[int(c.index[c.ID.eq(cid)][0]),j] for cid in selected.ID],default=np.nan) for j in range(len(z))]; cov['Covered_demand_weight']=np.where(covered,cov['Demand_weight_illustrative'],0)
        cov.to_csv(OUT/f'zone_coverage_{label.lower()}.csv',index=False)
    comp=pd.DataFrame(summary); comp.to_csv(OUT/'budget_comparison.csv',index=False)
    # Charts: each chart is generated from computed outputs.
    plt.figure(figsize=(10,6)); rr=ranked.sort_values('Priority_score'); plt.barh(rr['Candidate area'],rr['Priority_score']); plt.xlabel('Priority score (0–100)'); plt.title('Candidate Location Priority Ranking'); plt.tight_layout(); plt.savefig(OUT/'01_candidate_priority_ranking.png',dpi=170); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(comp['Budget_scenario'],comp['Weighted_coverage_percent_proxy'],marker='o',label='Demand-weighted coverage proxy'); plt.plot(comp['Budget_scenario'],comp['Zone_count_coverage_percent_proxy'],marker='s',label='Zone-count coverage proxy'); plt.ylabel('Coverage (%)'); plt.ylim(0,100); plt.title('Coverage Proxy by Budget Scenario'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'02_budget_vs_coverage.png',dpi=170); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(comp['Budget_scenario'],comp['Cost_used_units'],label='Cost used'); plt.plot(comp['Budget_scenario'],comp['Budget_units_illustrative'],marker='o',label='Available budget'); plt.ylabel('Illustrative cost units'); plt.title('Budget Use by Scenario'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'03_budget_utilization.png',dpi=170); plt.close()
    plt.figure(figsize=(10,6)); plt.scatter(z['X_km_illustrative'],z['Y_km_illustrative'],s=z['Demand_weight_illustrative']*2,c=z['Demand_weight_illustrative'],marker='o',label='Demand zones'); plt.scatter(c['X_km_illustrative'],c['Y_km_illustrative'],marker='^',s=110,label='Candidate stations');
    for _,r in z.iterrows(): plt.annotate(r['Zone_ID'],(r['X_km_illustrative'],r['Y_km_illustrative']),fontsize=7,xytext=(3,3),textcoords='offset points')
    for _,r in c.iterrows(): plt.annotate(r['ID'],(r['X_km_illustrative'],r['Y_km_illustrative']),fontsize=8,xytext=(3,-8),textcoords='offset points')
    plt.xlabel('Illustrative local X (km)'); plt.ylabel('Illustrative local Y (km)'); plt.title('Illustrative Demand Zones and Candidate Locations'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'04_demand_and_candidates_map.png',dpi=170); plt.close()
    # coverage status by zone across scenarios
    covmat=np.array([[scenario_cover[l][j] for j in range(len(z))] for l in labels],dtype=int)
    plt.figure(figsize=(11,4)); plt.imshow(covmat,aspect='auto',interpolation='nearest',vmin=0,vmax=1); plt.xticks(range(len(z)),z['Zone_ID'],rotation=0); plt.yticks(range(len(labels)),labels); plt.xlabel('Demand zone'); plt.title('Demand-Zone Coverage by Budget Scenario (1 = covered)'); plt.colorbar(label='Covered'); plt.tight_layout(); plt.savefig(OUT/'05_zone_coverage_matrix.png',dpi=170); plt.close()
    plt.figure(figsize=(9,5)); vals=[z.loc[scenario_cover[l],'Demand_weight_illustrative'].sum() for l in labels]; un=[z.loc[~scenario_cover[l],'Demand_weight_illustrative'].sum() for l in labels]; plt.bar(labels,vals,label='Covered demand weight'); plt.bar(labels,un,bottom=vals,label='Uncovered demand weight'); plt.ylabel('Illustrative demand weight'); plt.title('Covered vs Uncovered Demand Weight'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'06_covered_uncovered_demand.png',dpi=170); plt.close()
    plt.figure(figsize=(8,5)); counts=ranked['Grid capacity'].value_counts().reindex(['High','Medium','Low'],fill_value=0); plt.bar(counts.index,counts.values); plt.ylabel('Number of candidate sites'); plt.title('Candidate Sites by Grid-Capacity Category'); plt.tight_layout(); plt.savefig(OUT/'07_grid_capacity_feasibility.png',dpi=170); plt.close()
    # Machine-readable run summary and human-readable summary
    with open(OUT/'results_summary.txt','w',encoding='utf-8') as f:
        f.write('OPTIMAL EV CHARGING STATION PLACEMENT — RUN SUMMARY\n')
        f.write('='*55+'\n\n')
        f.write('IMPORTANT: Dataset positions, demand weights, costs and outputs are illustrative/simulated.\n')
        f.write('Coverage is based on straight-line distance in an illustrative local coordinate grid, not road routes.\n')
        f.write(f'Coverage radius used: {radius:.1f} illustrative km\n')
        f.write(f'Candidate sites: {len(c)} | Demand zones: {len(z)}\n\n')
        f.write('Candidate ranking (top to bottom):\n')
        f.write(ranked[['ID','Candidate area','Priority_score','Score_per_cost','Grid_feasible']].to_string(index=False)+'\n\n')
        f.write('Budget scenarios:\n'+comp[['Budget_scenario','Budget_units_illustrative','Stations_selected','Cost_used_units','Weighted_coverage_percent_proxy','Utilization_percent_proxy','Selected_areas']].to_string(index=False)+'\n')
    print('\n=== TOP CANDIDATE RANKING ===')
    print(ranked[['ID','Candidate area','Priority_score','Score_per_cost','Grid_feasible']].head(12).to_string(index=False))
    print('\n=== BUDGET SCENARIOS ===')
    print(comp[['Budget_scenario','Budget_units_illustrative','Stations_selected','Cost_used_units','Demand_zones_covered','Weighted_coverage_percent_proxy','Utilization_percent_proxy','Selected_areas']].to_string(index=False))
    print(f'\nCreated outputs in: {OUT}')
    print('NOTE: All inputs and metrics are illustrative; replace with verified local data for real-world claims.')

def main():
    p=argparse.ArgumentParser(description='Rank candidate EV charging sites and compare budget scenarios.')
    p.add_argument('--candidates',type=Path,default=DATA/'candidate_locations.csv')
    p.add_argument('--zones',type=Path,default=DATA/'demand_zones_illustrative.csv')
    p.add_argument('--budgets',nargs='+',type=float,default=[35,65,110])
    p.add_argument('--radius',type=float,default=RADIUS_KM,help='Illustrative straight-line coverage radius in km')
    a=p.parse_args()
    if a.radius<=0: p.error('--radius must be positive')
    c,z=load_inputs(a.candidates,a.zones)
    make_outputs(c,z,a.budgets,a.radius)
if __name__=='__main__': main()
