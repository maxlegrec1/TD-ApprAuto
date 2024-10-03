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


| Statistics                                 | mean          | std           | median        | min           | max           | 
|--------------------------------------------|---------------|---------------|---------------|---------------|---------------|
| Carbon concentration                       | 0.0755215     | 0.0238981     | 0.074         | 0.029         | 0.18          |               
| Silicon concentration                      | 0.328577      | 0.112455      | 0.32          | 0.04          | 1.14          |               
| Manganese concentration                    | 1.20282       | 0.382137      | 1.27          | 0.27          | 2.25          |               
| Sulphur concentration                      | 0.00952913    | 0.0112263     | 0.007         | 0.001         | 0.14          |               
| Phosphorus concentration                   | 0.0129525     | 0.0196268     | 0.01          | 0.002         | 0.25          |               
| Nickel concentration                       | 0.415034      | 0.786951      | 0.067         | 0             | 3.5           |               
| Chromium concentration                     | 2.10127       | 3.02655       | 0.53          | 0             | 10.2          |               
| Molybdenum concentration                   | 0.479172      | 0.477404      | 0.34          | 0             | 1.5           |               
| Vanadium concentration                     | 0.0973789     | 0.492889      | 0.0095        | 0             | 5             |               
| Copper concentration                       | 0.172163      | 0.322933      | 0.03          | 0             | 1.63          |               
| Cobalt concentration                       | 0.0710233     | 0.353735      | 0.005         | 0             | 2.8           |               
| Tungsten concentration                     | 0.134267      | 0.454958      | 0             | 0             | 2.99          |               
| Oxygen concentration                       | 441.967       | 147.484       | 423           | 132           | 1650          |               
| Titanium concentration                     | 80.5643       | 97.0864       | 42            | 0             | 690           |               
| Nitrogen concentration                     | 110.709       | 94.3686       | 82            | 21            | 552           |               
| Aluminium concentration                    | 102.347       | 141.474       | 44            | 0.004         | 680           |               
| Boron concentration                        | 7.87897       | 9.54217       | 5             | 1             | 69            |               
| Niobium concentration                      | 134.036       | 228.687       | 5             | 0             | 1000          |               
| Tin concentration                          | 42.171        | 89.0751       | 40            | 0             | 1000          |               
| Arsenic concentration                      | 31.0447       | 43.003        | 10            | 0.003         | 200           |               
| Antimony concentration                     | 29.4561       | 36.7828       | 15            | 0             | 200           |               
| Current                                    | 283.844       | 192.561       | 170           | 115           | 900           |               
| Voltage                                    | 27.6074       | 12.5556       | 21            | 11.5          | 75.36         |               
| Heat input                                 | 1.70099       | 1.29846       | 1             | 0.6           | 7.9           |               
| Interpass temperature                      | 204.215       | 39.3491       | 200           | 20            | 300           |               
| Post weld heat treatment temperature       | 304.674       | 285.498       | 250           | 0             | 760           |               
| Post weld heat treatment time              | 5.04965       | 6.09603       | 2             | 0             | 24            |               
| Yield strength                             | 508.557       | 92.8654       | 495           | 315           | 920           |               
| Ultimate tensile strength                  | 594.386       | 88.6362       | 575.5         | 447           | 1151          |               
| Elongation                                 | 26.2757       | 4.89599       | 26.8          | 10.6          | 37            |               
| Reduction of Area                          | 71.7999       | 8.92655       | 75            | 17            | 83            |               
| Charpy temperature                         | -34.6064      | 34.7386       | -40           | -114          | 188           |               
| Charpy impact toughness                    | 87.6894       | 50.1167       | 100           | 3             | 270           |               
| Hardness                                   | 226.906       | 57.7484       | 224           | 143           | 467           |              
| 50 % FATT                                  | -31.0968      | 43.6443       | -15           | -126          | 30            |               
| Primary ferrite in microstructure          | 19.1735       | 10.9827       | 19            | 0             | 48            |               
| Ferrite with second phase                  | 25.9556       | 21.2835       | 18            | 3             | 100           |              
| Acicular ferrite                           | 52.8333       | 23.4838       | 60            | 0             |               |              
| Martensite                                 | 0.337079      | 3.17999       | 0             | 0             |               |              
| Ferrite with carbide aggregate             | 0.438202      | 1.39769       | 0             | 0             |               |              

