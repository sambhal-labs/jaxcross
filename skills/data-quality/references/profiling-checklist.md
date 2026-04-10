# Data Profiling Checklist

## Schema Checks (Critical)
1. No duplicate column names
2. No entirely empty columns (all NaN)
3. No entirely empty rows
4. No mixed-type columns (strings + numbers in same column)
5. Column count matches expected schema (if known)
6. No whitespace-only string values

## Missing Data Checks
7. Per-column missing rate calculated
8. Columns with >50% missing flagged for potential removal
9. Rows with >80% missing flagged
10. Correlated missingness checked (MNAR indicator)
11. Missing patterns visualized or described (monotone, random, block)
12. Literal "null"/"nan"/"N/A"/""/" " strings detected and converted

## Numeric Distribution Checks
13. Min/max/mean/median/std computed per column
14. Skewness computed (flag if |skew| > 2)
15. Outliers detected (>3 sigma or IQR method)
16. Infinite values detected (`np.inf`, `-np.inf`)
17. Negative values in columns expected to be non-negative
18. Zero-inflation checked (>50% zeros)
19. Scale consistency across similar columns (e.g., all monetary columns in same unit)

## Categorical Distribution Checks
20. Cardinality computed per column
21. High cardinality flagged (>100 unique for categorical)
22. Rare categories flagged (<5 occurrences)
23. Case inconsistency detected ("Male" vs "male" vs "MALE")
24. Leading/trailing whitespace in string values
25. Single-category dominance (>95% one value)

## Binary Column Checks
26. Class balance computed
27. Severe imbalance flagged (<5% or >95%)
28. Non-{0,1} values in expected binary columns

## Temporal Checks (if dates present)
29. Date range sanity (no future dates if not expected)
30. Gaps in time series detected
31. Timezone consistency
32. Date format consistency within column

## Cross-Column Checks
33. Pairwise correlations computed for numeric columns
34. Perfect correlation detected (|r| > 0.99) — possible duplicates
35. ID-like columns detected (unique per row, sequential)
36. Target leakage candidates identified (columns with suspiciously high correlation to target)

## jaxcross Compatibility Checks
37. All columns can be cast to float32 (after encoding)
38. Category values are 0-indexed contiguous integers
39. Cyclic candidates identified (hour, angle, compass)
40. Ordinal candidates identified (ratings, education levels)
41. `max_categories` requirement estimated per categorical column
