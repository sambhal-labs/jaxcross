# Mutual Information

## What

Estimate mutual information (MI) between pairs of columns — a continuous measure of how much knowing one column tells you about another. Also provides Linfoot correlation, a normalized version.

## When to Use

- Measuring dependency strength (not just presence)
- Feature selection — which features carry the most information about a target
- Comparing dependency strengths across column pairs

## Basic Usage

```python
from crosscat import mutual_information

mi, linfoot = mutual_information([state], col_i=0, col_j=1)
print(f"MI(salary, experience): {mi:.3f}")
print(f"Linfoot correlation: {linfoot:.3f}")  # normalized to [0, 1]
```

## Comparing Dependencies

```python
# Which column is more informative about salary?
mi_exp, _ = mutual_information([state], col_i=0, col_j=1)    # salary, experience
mi_dept, _ = mutual_information([state], col_i=0, col_j=2)   # salary, department
mi_remote, _ = mutual_information([state], col_i=0, col_j=3) # salary, is_remote

print(f"MI(salary, experience): {mi_exp:.3f}")
print(f"MI(salary, department): {mi_dept:.3f}")
print(f"MI(salary, is_remote):  {mi_remote:.3f}")
```

## MI vs Dependence Probability

| Measure | What It Tells You | Range |
|---------|------------------|-------|
| `dependence_probability` | Whether columns are in the same view (structural) | [0, 1] |
| `mutual_information` | How much information they share (quantitative) | [0, inf) |
| Linfoot correlation | Normalized MI | [0, 1] |

Columns can have high dependence probability (always in same view) but low MI if the relationship is weak.

## Conditional Entropy

How much uncertainty remains about a target column given other columns:

```python
from crosscat import conditional_entropy

h = conditional_entropy(key, [state], data, target_col=0, given_cols=[1, 2])
print(f"H(salary | experience, department): {h:.3f} nats")
```

## Packed Versions

```python
from crosscat import packed_mutual_information, packed_conditional_entropy

mi, linfoot = packed_mutual_information([packed], col_types, col_i=0, col_j=1)
h = packed_conditional_entropy(key, [packed], data, target_col=0, given_cols=[1, 2])
```

## API Reference

- [`mutual_information`](../../api/inference.md#mutual_information)
- [`conditional_entropy`](../../api/inference.md#conditional_entropy)
