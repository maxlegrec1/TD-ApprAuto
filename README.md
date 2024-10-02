# Requirements
```pip install pandas numpy torch sklearn```

# Columns :
| Name                                 | Type    | Not N | Valid | To convert | Comment                                                |
|--------------------------------------|---------|-------|-------|------------|--------------------------------------------------------|
| Carbon concentration                 | float64 | 1652  | 1652  | 0          |                                                        |
| Silicon concentration                | float64 | 1652  | 1652  | 0          |                                                        |
| Manganese concentration              | float64 | 1652  | 1652  | 0          |                                                        |
| Sulphur concentration                | float64 | 1648  | 1648  | 0          |                                                        |
| Phosphorus concentration             | float64 | 1642  | 1642  | 0          |                                                        |
| Nickel concentration                 | float64 | 697   | 697   | 0          |                                                        |
| Chromium concentration               | float64 | 784   | 784   | 0          |                                                        |
| Molybdenum concentration             | float64 | 793   | 793   | 0          |                                                        |
| Vanadium concentration               | float64 | 928   | 928   | 0          |                                                        |
| Copper concentration                 | float64 | 578   | 578   | 0          |                                                        |
| Cobalt concentration                 | float64 | 129   | 129   | 0          |                                                        |
| Tungsten concentration               | float64 | 75    | 75    | 0          |                                                        |
| Oxygen concentration                 | float64 | 1256  | 1256  | 0          |                                                        |
| Titanium concentration               | float64 | 935   | 935   | 0          |                                                        |
| Nitrogen concentration               | float64 | 1242  | 1183  | 59         | 59 : "\d+tot(\d+\|nd)res" -> 67tot33res                |
| Aluminium concentration              | float64 | 905   | 905   | 0          |                                                        |
| Boron concentration                  | float64 | 504   | 504   | 0          |                                                        |
| Niobium concentration                | float64 | 752   | 752   | 0          |                                                        |
| Tin concentration                    | float64 | 296   | 296   | 0          |                                                        |
| Arsenic concentration                | float64 | 234   | 234   | 0          |                                                        |
| Antimony concentration               | float64 | 260   | 260   | 0          |                                                        |
| Current                              | float64 | 1404  | 1404  | 0          |                                                        |
| Voltage                              | float64 | 1404  | 1404  | 0          |                                                        |
| AC or DC                             | string  | 1437  | 1437  | 0          | AC, DC                                                 |
| Electrode positive or negative       | string  | 1496  | 1458  | 38         | Some are neither '+', '-' nor 'N' but 0                |
| Heat input                           | float64 | 1652  | 1652  | 0          |                                                        |
| Interpass temperature                | float64 | 1652  | 1614  | 38         | 38 : "\d+-\d+" -> 150-200                              |
| Type of weld                         | string  | 1652  | 1652  | 0          | MMA, SA, FCA, TSA, ShMA, NGSAW, NGGMA, SAA, GTAA, GMAA |
| Post weld heat treatment temperature | float64 | 1639  | 1639  | 0          |                                                        |
| Post weld heat treatment time        | float64 | 1639  | 1639  | 0          |                                                        |
| Yield strength                       | float64 | 780   | 780   | 0          |                                                        |
| Ultimate tensile strength            | float64 | 738   | 738   | 0          |                                                        |
| Elongation                           | float64 | 700   | 700   | 0          |                                                        |
| Reduction of Area                    | float64 | 705   | 705   | 0          |                                                        |
| Charpy temperature                   | float64 | 879   | 879   | 0          |                                                        |
| Charpy impact toughness              | float64 | 879   | 879   | 0          |                                                        |
| Hardness                             | float64 | 138   | 80    | 58         | 58 : "\d+\(?Hv\d+\)?" -> 203(Hv30)                     |
| 50 % FATT                            | float64 | 31    | 31    | 0          |                                                        |
| Primary ferrite in microstructure    | float64 | 98    | 98    | 0          |                                                        |
| Ferrite with second phase            | float64 | 90    | 90    | 0          |                                                        |
| Acicular ferrite                     | float64 | 90    | 90    | 0          |                                                        |
| Martensite                           | float64 | 89    | 89    | 0          |                                                        |
| Ferrite with carbide aggreagate      | float64 | 89    | 89    | 0          |                                                        |
| Weld ID                              | string  | 1652  | 1652  | 0          | 1490 categories                                        |
