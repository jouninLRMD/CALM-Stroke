# CALM-Stroke results (auto-generated)


## Scheme: cv5 (n = 718 patients, models with complete predictions)

```
                                      n    auc  auc_lo  auc_hi  dAUC_vs_R0_clinical  dAUC_vs_R0_clinical_lo  dAUC_vs_R0_clinical_hi  dAUC_vs_R1b_clin_vol_chronic_radc  dAUC_vs_R1b_clin_vol_chronic_radc_lo  dAUC_vs_R1b_clin_vol_chronic_radc_hi  brier  cal_slope  ordinal_c
model                                                                                                                                                                                                                                                                          
calm25d_noaux                       718  0.837   0.808   0.867                0.081                   0.058                   0.105                              0.038                                 0.015                                 0.062  0.165      1.056      0.759
calm25d                             718  0.827   0.797   0.857                0.071                   0.051                   0.092                              0.028                                 0.009                                 0.049  0.171      1.081      0.754
calm25d_noseg                       718  0.826   0.797   0.857                0.070                   0.048                   0.093                              0.028                                 0.006                                 0.050  0.171      1.039      0.753
slice25d                            718  0.826   0.795   0.855                0.070                   0.045                   0.094                              0.027                                 0.006                                 0.050  0.169      1.028      0.756
slice25d_lesanchor                  718  0.818   0.787   0.850                0.063                   0.035                   0.091                              0.020                                 0.010                                 0.030  0.174      0.950      0.751
calm25d_noanchor                    718  0.811   0.779   0.842                0.055                   0.017                   0.094                              0.012                                -0.016                                 0.042  0.177      1.263      0.747
calm                                718  0.805   0.774   0.835                0.049                   0.024                   0.074                              0.007                                -0.013                                 0.026  0.180      1.023      0.736
baseline_R1b_clin_vol_chronic_radc  718  0.798   0.766   0.831                0.043                   0.016                   0.070                              0.000                                 0.000                                 0.000  0.181      0.929      0.732
baseline_R1_clin_volume             718  0.795   0.763   0.827                0.040                   0.014                   0.066                             -0.003                                -0.012                                 0.005  0.183      0.949      0.728
baseline_R2b_clin_vol_topography    718  0.794   0.762   0.825                0.038                   0.011                   0.063                             -0.005                                -0.019                                 0.009  0.184      0.913      0.728
baseline_R2a_clin_vol_territory     718  0.786   0.753   0.820                0.031                   0.004                   0.057                             -0.012                                -0.025                                 0.000  0.187      0.882      0.725
resnet3d                            718  0.778   0.745   0.811                0.023                  -0.003                   0.047                             -0.020                                -0.042                                 0.002  0.192      0.795      0.714
baseline_R0_clinical                718  0.756   0.721   0.791                0.000                   0.000                   0.000                             -0.043                                -0.070                                -0.016  0.203      0.957      0.692
resnet3d_noanchor                   718  0.740   0.702   0.775               -0.015                  -0.060                   0.028                             -0.058                                -0.093                                -0.025  0.205      0.902      0.694
baseline_I1_volume_only             718  0.739   0.701   0.775               -0.017                  -0.062                   0.027                             -0.060                                -0.086                                -0.032  0.205      0.976      0.686
baseline_I2_topography_only         718  0.724   0.683   0.762               -0.032                  -0.077                   0.014                             -0.075                                -0.108                                -0.043  0.207      1.048      0.678
```


### Subgroup AUC (cv5)

```
                         n  n_poor  baseline_I1_volume_only  baseline_I2_topography_only  baseline_R0_clinical  baseline_R1_clin_volume  baseline_R1b_clin_vol_chronic_radc  baseline_R2a_clin_vol_territory  baseline_R2b_clin_vol_topography  calm25d  calm25d_noanchor  calm25d_noaux  calm25d_noseg   calm  resnet3d  resnet3d_noanchor  slice25d  slice25d_lesanchor
subgroup                                                                                                                                                                                                                                                                                                                                                                 
1.5 T                  540     257                    0.722                        0.703                 0.746                    0.780                               0.788                            0.767                             0.783    0.821             0.807          0.837          0.822  0.791     0.767              0.731     0.819               0.811
3 T                    178      80                    0.780                        0.777                 0.782                    0.839                               0.823                            0.842                             0.823    0.839             0.820          0.830          0.840  0.846     0.814              0.769     0.844               0.836
female                 335     152                    0.751                        0.721                 0.781                    0.820                               0.817                            0.806                             0.817    0.838             0.819          0.856          0.835  0.831     0.800              0.739     0.826               0.838
male                   383     185                    0.728                        0.726                 0.733                    0.773                               0.781                            0.769                             0.773    0.817             0.802          0.818          0.818  0.781     0.760              0.741     0.825               0.801
Black                  151      78                    0.687                        0.702                 0.729                    0.762                               0.758                            0.759                             0.765    0.779             0.808          0.798          0.784  0.765     0.741              0.703     0.794               0.790
White                  526     237                    0.749                        0.725                 0.777                    0.812                               0.817                            0.800                             0.809    0.848             0.814          0.849          0.841  0.826     0.794              0.752     0.837               0.832
age >= 80               82      51                    0.679                        0.669                 0.784                    0.762                               0.767                            0.740                             0.772    0.824             0.746          0.837          0.782  0.781     0.722              0.603     0.800               0.772
age < 80               636     286                    0.752                        0.737                 0.749                    0.795                               0.798                            0.787                             0.793    0.825             0.815          0.835          0.828  0.804     0.779              0.751     0.826               0.820
posterior-predominant  163      71                    0.682                        0.615                 0.739                    0.767                               0.755                            0.744                             0.748    0.777             0.736          0.772          0.785  0.766     0.693              0.619     0.771               0.769
anterior-predominant   443     240                    0.752                        0.743                 0.748                    0.797                               0.806                            0.789                             0.805    0.834             0.830          0.852          0.829  0.804     0.796              0.766     0.838               0.826
left-predominant       321     171                    0.722                        0.700                 0.745                    0.769                               0.776                            0.761                             0.765    0.801             0.768          0.806          0.792  0.793     0.752              0.696     0.793               0.790
right-predominant      285     140                    0.760                        0.734                 0.755                    0.816                               0.820                            0.795                             0.818    0.847             0.855          0.868          0.857  0.806     0.804              0.773     0.863               0.839
no expert mask         112      26                    0.338                        0.338                 0.695                    0.663                               0.663                            0.666                             0.674    0.766             0.682          0.755          0.754  0.739     0.675              0.659     0.724               0.738
```


### AUC per fold and seed (cv5)

```
fold                                     fold0  fold1  fold2  fold3  fold4
model                              seed                                   
baseline_I1_volume_only            s0    0.689  0.763  0.742  0.739  0.771
baseline_I2_topography_only        s0    0.697  0.710  0.784  0.719  0.735
baseline_R0_clinical               s0    0.723  0.828  0.746  0.739  0.747
baseline_R1_clin_volume            s0    0.748  0.835  0.798  0.802  0.807
baseline_R1b_clin_vol_chronic_radc s0    0.759  0.828  0.789  0.805  0.816
baseline_R2a_clin_vol_territory    s0    0.740  0.832  0.770  0.785  0.818
baseline_R2b_clin_vol_topography   s0    0.753  0.805  0.801  0.805  0.828
calm                               s0    0.730  0.848  0.795  0.815  0.777
                                   s1    0.765  0.832  0.789  0.828  0.815
                                   s2    0.772  0.814  0.794  0.799  0.817
calm25d                            s0    0.777  0.850  0.812  0.814  0.835
                                   s1    0.792  0.864  0.804  0.798  0.788
                                   s2    0.768  0.893  0.752  0.799  0.815
                                   s3    0.766  0.867  0.851  0.821  0.799
                                   s4    0.764  0.891  0.780  0.763  0.806
calm25d_noanchor                   s0    0.732  0.841  0.782  0.770  0.816
                                   s1    0.719  0.810  0.782  0.817  0.776
                                   s2    0.713  0.793  0.824  0.851  0.860
calm25d_noaux                      s0    0.789  0.825  0.821  0.841  0.845
                                   s1    0.767  0.880  0.824  0.830  0.825
                                   s2    0.779  0.882  0.796  0.834  0.803
calm25d_noseg                      s0    0.782  0.848  0.825  0.807  0.836
                                   s1    0.808  0.856  0.796  0.833  0.803
                                   s2    0.735  0.852  0.797  0.832  0.786
resnet3d                           s0    0.739  0.819  0.804  0.769  0.807
resnet3d_noanchor                  s0    0.734  0.836  0.781  0.767  0.725
slice25d                           s0    0.796  0.860  0.827  0.795  0.830
                                   s1    0.767  0.889  0.798  0.805  0.822
                                   s2    0.755  0.873  0.817  0.827  0.788
                                   s3    0.788  0.878  0.782  0.786  0.819
                                   s4    0.799  0.860  0.817  0.799  0.803
slice25d_lesanchor                 s0    0.771  0.835  0.797  0.816  0.859
                                   s1    0.794  0.848  0.799  0.826  0.824
                                   s2    0.776  0.817  0.789  0.808  0.812
                                   s3    0.770  0.831  0.792  0.824  0.820
                                   s4    0.763  0.840  0.813  0.783  0.836
```


### Net benefit at selected thresholds (cv5)

```
           treat all  treat none  calm25d_noaux  calm25d  calm25d_noseg  slice25d  slice25d_lesanchor  calm25d_noanchor   calm  R1b_clin_vol_chronic_radc  R2b_clin_vol_topography  resnet3d  R0_clinical  resnet3d_noanchor
threshold                                                                                                                                                                                                                   
0.2            0.337         0.0          0.355    0.357          0.363     0.350               0.355             0.343  0.353                      0.344                    0.347     0.340        0.339              0.340
0.3            0.242         0.0          0.314    0.309          0.308     0.308               0.302             0.291  0.298                      0.288                    0.288     0.283        0.279              0.262
0.4            0.116         0.0          0.272    0.269          0.265     0.275               0.263             0.250  0.253                      0.246                    0.243     0.236        0.242              0.207
0.5           -0.061         0.0          0.234    0.221          0.223     0.230               0.228             0.213  0.191                      0.206                    0.192     0.185        0.167              0.150
0.6           -0.327         0.0          0.187    0.155          0.166     0.180               0.170             0.178  0.161                      0.164                    0.169     0.148        0.098              0.108
```


## Scheme: loso (n = 718 patients, models with complete predictions)

```
                                      n    auc  auc_lo  auc_hi  dAUC_vs_R0_clinical  dAUC_vs_R0_clinical_lo  dAUC_vs_R0_clinical_hi  dAUC_vs_R1b_clin_vol_chronic_radc  dAUC_vs_R1b_clin_vol_chronic_radc_lo  dAUC_vs_R1b_clin_vol_chronic_radc_hi  brier  cal_slope  ordinal_c
model                                                                                                                                                                                                                                                                          
calm25d                             718  0.826   0.796   0.856                0.073                   0.052                   0.094                              0.032                                 0.012                                 0.054  0.169      0.960      0.751
slice25d                            718  0.817   0.786   0.849                0.063                   0.041                   0.087                              0.023                                 0.002                                 0.046  0.176      0.959      0.743
slice25d_lesanchor                  718  0.802   0.770   0.834                0.048                   0.021                   0.078                              0.008                                -0.005                                 0.021  0.181      0.886      0.739
baseline_R2b_clin_vol_topography    718  0.798   0.765   0.829                0.044                   0.017                   0.071                              0.004                                -0.010                                 0.017  0.182      0.989      0.729
baseline_R1_clin_volume             718  0.798   0.765   0.829                0.044                   0.018                   0.071                              0.004                                -0.004                                 0.011  0.182      0.944      0.728
calm                                718  0.798   0.766   0.831                0.044                   0.018                   0.070                              0.003                                -0.018                                 0.027  0.188      0.870      0.721
baseline_R1b_clin_vol_chronic_radc  718  0.794   0.762   0.827                0.040                   0.012                   0.069                              0.000                                 0.000                                 0.000  0.183      0.907      0.727
baseline_R2a_clin_vol_territory     718  0.784   0.751   0.818                0.030                   0.004                   0.058                             -0.010                                -0.022                                 0.003  0.188      0.857      0.721
baseline_R0_clinical                718  0.754   0.719   0.789                0.000                   0.000                   0.000                             -0.040                                -0.069                                -0.012  0.204      0.940      0.690
resnet3d                            718  0.750   0.714   0.786               -0.004                  -0.027                   0.020                             -0.044                                -0.071                                -0.016  0.206      0.757      0.685
baseline_I1_volume_only             718  0.738   0.701   0.775               -0.015                  -0.061                   0.032                             -0.056                                -0.081                                -0.029  0.205      0.950      0.687
baseline_I2_topography_only         718  0.719   0.679   0.756               -0.035                  -0.080                   0.010                             -0.076                                -0.109                                -0.042  0.208      1.052      0.672
```


### AUC per held-out scanner

```
held_out_scanner                    Achieva  Ingenia Ambition X  Ingenia Elition X  Intera
model                                                                                     
baseline_I1_volume_only               0.764               0.721              0.768   0.740
baseline_I2_topography_only           0.731               0.699              0.760   0.808
baseline_R0_clinical                  0.772               0.736              0.775   0.776
baseline_R1_clin_volume               0.826               0.779              0.831   0.801
baseline_R1b_clin_vol_chronic_radc    0.817               0.778              0.804   0.815
baseline_R2a_clin_vol_territory       0.813               0.760              0.823   0.812
baseline_R2b_clin_vol_topography      0.823               0.784              0.798   0.848
calm                                  0.811               0.808              0.825   0.787
calm25d                               0.838               0.818              0.828   0.857
resnet3d                              0.769               0.770              0.773   0.796
slice25d                              0.830               0.807              0.809   0.846
slice25d_lesanchor                    0.832               0.778              0.812   0.821
```


## Scheme: field (n = 178 patients, models with complete predictions)

```
                                      n    auc  auc_lo  auc_hi  dAUC_vs_R0_clinical  dAUC_vs_R0_clinical_lo  dAUC_vs_R0_clinical_hi  dAUC_vs_R1b_clin_vol_chronic_radc  dAUC_vs_R1b_clin_vol_chronic_radc_lo  dAUC_vs_R1b_clin_vol_chronic_radc_hi  brier  cal_slope  ordinal_c
model                                                                                                                                                                                                                                                                          
calm                                178  0.865   0.808   0.914                0.085                   0.024                   0.146                              0.045                                -0.005                                 0.100  0.150      1.478      0.761
calm25d                             178  0.851   0.791   0.901                0.071                   0.023                   0.124                              0.031                                -0.024                                 0.089  0.167      1.216      0.757
slice25d                            178  0.850   0.792   0.903                0.069                   0.005                   0.135                              0.030                                -0.030                                 0.090  0.154      1.097      0.758
slice25d_lesanchor                  178  0.836   0.772   0.892                0.056                  -0.010                   0.122                              0.016                                -0.020                                 0.050  0.163      1.005      0.744
baseline_R1_clin_volume             178  0.832   0.768   0.892                0.051                  -0.005                   0.106                              0.012                                -0.012                                 0.042  0.164      1.296      0.739
baseline_R2a_clin_vol_territory     178  0.828   0.764   0.888                0.048                  -0.009                   0.105                              0.009                                -0.023                                 0.046  0.165      1.221      0.742
baseline_R1b_clin_vol_chronic_radc  178  0.820   0.752   0.883                0.039                  -0.027                   0.103                              0.000                                 0.000                                 0.000  0.167      1.117      0.735
baseline_R2b_clin_vol_topography    178  0.817   0.753   0.879                0.036                  -0.022                   0.096                             -0.003                                -0.037                                 0.036  0.169      1.066      0.743
baseline_I2_topography_only         178  0.802   0.733   0.864                0.022                  -0.064                   0.106                             -0.018                                -0.080                                 0.047  0.191      1.391      0.712
resnet3d                            178  0.796   0.725   0.860                0.015                  -0.036                   0.067                             -0.024                                -0.078                                 0.029  0.184      1.299      0.715
baseline_R0_clinical                178  0.780   0.714   0.849                0.000                   0.000                   0.000                             -0.039                                -0.103                                 0.027  0.192      1.435      0.705
baseline_I1_volume_only             178  0.766   0.688   0.834               -0.014                  -0.103                   0.069                             -0.054                                -0.103                                -0.005  0.188      1.296      0.689
```
