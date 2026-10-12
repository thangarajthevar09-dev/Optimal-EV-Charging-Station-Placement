# Methodology and formulas

## Candidate priority score
Category levels are encoded Low=1, Medium=2, High=3.

Priority = 100 × [0.20(Traffic/3) + 0.35(TravelDemand/3) + 0.30((4−ExistingCoverage)/3) + 0.15(GridCapacity/3)].

Higher traffic and demand raise priority. Lower existing coverage raises priority. Higher grid capacity raises feasibility/priority. A low-capacity site is excluded from selection pending a grid review.

## Distance-based coverage proxy
Each candidate and demand zone has an illustrative local X/Y coordinate in kilometres. Euclidean distance is calculated as sqrt((Xc−Xz)^2+(Yc−Yz)^2). A zone is covered if at least one selected site is within the configured radius (default 2.0 illustrative km). This is not a road-network distance model.

## Greedy budget selection
For each feasible unselected site, calculate the demand weight of currently uncovered zones that would newly fall within its radius. Add a small priority-score tie-break contribution (0.20 × priority score), divide by site cost, and select the highest benefit-per-cost site that fits the remaining budget. Repeat until no remaining site can be added. Greedy selection is a heuristic and does not guarantee a globally optimal solution.

## Metrics
- Weighted coverage proxy = demand weight of covered zones / total demand weight × 100.
- Zone-count coverage proxy = covered zones / all zones × 100.
- Utilization proxy = 35 + (mean selected travel-demand category score / 3 × 50), capped conceptually at 85 in this prototype. It is not actual charger utilization.

## Complexity
With C candidate sites and Z demand zones, distance construction is O(CZ). The greedy process recalculates marginal coverage for candidate sites at each selection, with a simple upper-bound around O(C²Z), suitable for a small academic dataset.
