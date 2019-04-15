import argparse
import collections
import logging
import io
import itertools

import hoomd
import flowws
from flowws import Argument as Arg
import numpy as np

logger = logging.getLogger(__name__)

@flowws.add_stage_arguments
class Damasceno2017Interaction(flowws.Stage):
    """Specify a new interaction potential from the paper "Non-close-packed three-dimensional quasicrystals" to include in future MD stages

    These interactions are taken directly from the supplemental
    information of the paper (Journal of Physics: Condensed Matter,
    Volume 29, Number 23; DOI 10.1088/1361-648X/aa6cc1).

    The SI provides 10 potentials with different well depth epsilon;
    currently this stage will select the reported potential with the
    nearest well depth to the given value.

    """
    ARGS = [
        Arg('reset', '-r', bool, False,
            help='Clear previously-defined interactions beforehand'),
        Arg('depth', '-d', float, required=True,
            help='Well depth epsilon (the nearest reported potential is used)'),
    ]

    def run(self, scope, storage):
        """Registers this object to provide a force compute in future MD stages"""
        callbacks = scope.setdefault('callbacks', collections.defaultdict(list))

        if self.arguments['reset']:
            pre_run_callbacks = [c for c in callbacks['pre_run']
                                 if not isinstance(c, Interaction)]
            callbacks['pre_run'] = pre_run_callbacks

        callbacks['pre_run'].append(self)

    def __call__(self, scope, storage, context):
        """Callback to be performed before each run command.

        Initializes a pair potential interaction based on per-type
        shape information.
        """
        nlist = hoomd.md.nlist.tree()
        system = scope['system']

        # subtract 1 to change index from from counting numbers (as
        # reported in the paper) to list indices (as used in this
        # module)
        potential_index = int(round((self.arguments['depth'] + 2.2)/0.4)) - 1
        if potential_index < 0 or potential_index >= len(SI_TABLE_POTENTIALS):
            logger.warning('Potential well depth out of range, using the closest one in range instead')
        potential_index = np.clip(potential_index, 0, 9)

        potential_table_data = SI_TABLE_POTENTIALS[potential_index]
        potential_table_data = io.StringIO(potential_table_data)

        potential_table = np.loadtxt(potential_table_data).reshape((-1, 3))
        rs, Us, Fs = potential_table.T
        width = len(rs)
        rmin, rmax = np.min(rs), np.max(rs)
        dr = (rmax - rmin)/(width - 1)

        def local_table_grabber(r, rmin, rmax):
            idx = int(np.round((r - rmin)/dr))
            return Us[idx], Fs[idx]

        table = hoomd.md.pair.table(width=width, nlist=nlist)
        all_types = list(system.particles.types)
        table.pair_coeff.set(
            all_types, all_types,
            func=local_table_grabber, rmin=rmin, rmax=rmax, coeff={})

# contents of "potential.01.dat" through "potential.10.dat",
# corresponding to a well depth of -2.2 + 0.4*i for
# "potential.i.dat". Each row contains "r, energy, force" data.
SI_TABLE_POTENTIALS = []

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83864	490.93773
0.32	126.25101	467.91539
0.34	117.11845	445.42915
0.36	108.43028	423.47738
0.38	100.17581	402.05816
0.4	92.344416	381.16932
0.42	84.925517	360.80839
0.44	77.908579	340.97274
0.46	71.283124	321.65953
0.48	65.038734	302.86591
0.5	59.165043	284.58909
0.52	53.651742	266.82653
0.54	48.488567	249.57613
0.56	43.665291	232.83648
0.58	39.171707	216.60707
0.6	34.997603	200.88862
0.62	31.132741	185.68327
0.64	27.566826	170.99482
0.66	24.289466	156.82893
0.68	21.290135	143.19321
0.7	18.558139	130.09723
0.72	16.082572	117.55241
0.74	13.852282	105.5718
0.76	11.855843	94.169712
0.78	10.081536	83.361221
0.8	8.5173357	73.161538
0.82	7.1509179	63.58531
0.84	5.9696778	54.64584
0.86	4.9607641	46.354303
0.88	4.11113	38.718996
0.9	3.4075968	31.744669
0.92	2.8369311	25.432008
0.94	2.3859292	19.777293
0.96	2.041507	14.772291
0.98	1.7907894	10.404385
1.	1.6211944	6.6569532
1.02	1.5205075	3.5099886
1.04	1.4769418	0.9409217
1.06	1.4791789	-1.0743984
1.08	1.5163906	-2.5606232
1.1	1.5782372	-3.5417931
1.12	1.6548449	-4.0402634
1.14	1.7367619	-4.0757363
1.16	1.8148969	-3.6644568
1.18	1.880444	-2.8186173
1.2	1.9253913	-1.6626294
1.22	1.9467745	-0.47177662
1.24	1.9440978	0.74390151
1.26	1.9168195	1.9895369
1.28	1.8642598	3.2737445
1.3	1.7855369	4.6079478
1.32	1.679519	6.0056206
1.34	1.5462424	7.2610051
1.36	1.3914707	8.1576386
1.38	1.2222284	8.7105069
1.4	1.0452472	8.9338151
1.42	0.86698652	8.8404658
1.44	0.69366419	8.4417065
1.46	0.53128295	7.7529135
1.48	0.38411531	6.9611986
1.5	0.25292481	6.1559525
1.52	0.13794282	5.3408098
1.54	0.039344829	4.5177579
1.56	-0.042718819	3.6873656
1.58	-0.1080971	2.8490389
1.6	-0.15665479	2.0141046
1.62	-0.18940797	1.2817212
1.64	-0.20873501	0.67103637
1.66	-0.21704208	0.17924771
1.68	-0.21667844	-0.19651031
1.7	-0.2099365	-0.45903829
1.72	-0.19905412	-0.6109777
1.74	-0.1861206	-0.67705052
1.76	-0.17200599	-0.73406984
1.78	-0.15676543	-0.7899375
1.8	-0.14040536	-0.84626675
1.82	-0.1229026	-0.90441225
1.84	-0.10420926	-0.96549191
1.86	-0.084257268	-1.0304133
1.88	-0.062978734	-1.0970974
1.9	-0.040387134	-1.1617574
1.92	-0.016519289	-1.2247865
1.94	0.0085957342	-1.2865243
1.96	0.034934799	-1.3472275
1.98	0.062479222	-1.4070873
2.	0.091213587	-1.4661997
2.02	0.12108848	-1.5198448
2.04	0.15194979	-1.5648506
2.06	0.18362526	-1.6012694
2.08	0.21594356	-1.6291377
2.1	0.24873394	-1.648481
2.12	0.28182609	-1.6593168
2.14	0.31504251	-1.6582531
2.16	0.34774369	-1.5999695
2.18	0.37856547	-1.4703133
2.2	0.40608043	-1.2692881
2.22	0.42886122	-0.99689606
2.24	0.4454805	-0.6531386
2.26	0.454511	-0.23801663
2.28	0.4546137	0.22449863
2.3	0.4460554	0.6170502
2.32	0.43050296	0.92391249
2.34	0.40967016	1.1450853
2.36	0.38527081	1.2805687
2.38	0.35901868	1.3303625
2.4	0.33262757	1.2944668
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83796	490.94933
0.32	126.25005	467.93118
0.34	117.11713	445.45045
0.36	108.42846	423.50582
0.38	100.17334	402.09577
0.4	92.341079	381.21854
0.42	84.921055	360.87218
0.44	77.902667	341.05456
0.46	71.275363	321.76343
0.48	65.028635	302.9965
0.5	59.152023	284.75154
0.52	53.635107	267.02652
0.54	48.467507	249.81976
0.56	43.63887	233.13012
0.58	39.138858	216.95721
0.6	34.957133	201.30159
0.62	31.083335	186.16498
0.64	27.507054	171.55043
0.66	24.217807	157.46245
0.68	21.205005	143.90714
0.7	18.45792	130.89215
0.72	15.965656	118.42662
0.74	13.717123	106.52098
0.76	11.701008	95.186699
0.78	9.9057654	84.435819
0.8	8.3196047	74.280516
0.82	6.9304959	64.732506
0.84	5.7261844	55.802433
0.86	4.6942189	47.499236
0.88	3.8219909	39.829553
0.9	3.0967873	32.797184
0.92	2.5058504	26.402676
0.94	2.0364468	20.643056
0.96	1.6759387	15.511736
0.98	1.4118553	10.998623
1.	1.2319605	7.0904182
1.02	1.1243121	3.7711174
1.04	1.0773098	1.0226646
1.06	1.0797285	-1.174261
1.08	1.120735	-2.8393805
1.1	1.1898869	-3.9919264
1.12	1.2771146	-4.6497827
1.14	1.372686	-4.8287114
1.16	1.4671576	-4.5417083
1.18	1.5513139	-3.7985275
1.2	1.6166937	-2.7220235
1.22	1.6598611	-1.5868265
1.24	1.6798446	-0.4031976
1.26	1.6756394	0.83296854
1.28	1.6461302	2.128564
1.3	1.5900401	3.4927274
1.32	1.5058923	4.9362379
1.34	1.3934347	6.2503904
1.36	1.2582027	7.215676
1.38	1.1070534	7.8440764
1.4	0.94660951	8.1469556
1.42	0.7832762	8.1346356
1.44	0.62326509	7.8161145
1.46	0.47261417	7.2048937
1.48	0.33566474	6.4866033
1.5	0.21327486	5.7495405
1.52	0.10578859	4.9966134
1.54	0.013505241	4.2294115
1.56	-0.063295953	3.4483903
1.58	-0.12433523	2.6530742
1.6	-0.1693529	1.8550896
1.62	-0.19924795	1.1540252
1.64	-0.21629116	0.56954349
1.66	-0.22279197	0.099402666
1.68	-0.22101425	-0.25868972
1.7	-0.21317641	-0.50697416
1.72	-0.20145321	-0.64756382
1.74	-0.18788101	-0.70469688
1.76	-0.17328605	-0.75475446
1.78	-0.15768779	-0.80526125
1.8	-0.14106396	-0.85750787
1.82	-0.12336861	-0.91257803
1.84	-0.10453602	-0.97136606
1.86	-0.084484306	-1.034598
1.88	-0.063135059	-1.1000498
1.9	-0.040493796	-1.1638203
1.92	-0.016591407	-1.2262141
1.94	0.0085474142	-1.2875028
1.96	0.034902717	-1.3478917
1.98	0.062458114	-1.407534
2.	0.091199825	-1.4664971
2.02	0.12107959	-1.520041
2.04	0.1519441	-1.5649788
2.06	0.18362165	-1.6013524
2.08	0.21594129	-1.6291909
2.1	0.24873253	-1.6485148
2.12	0.28182521	-1.659338
2.14	0.31504197	-1.6582663
2.16	0.34774337	-1.5999776
2.18	0.37856528	-1.4703183
2.2	0.40608032	-1.2692911
2.22	0.42886115	-0.99689788
2.24	0.44548046	-0.65313968
2.26	0.45451097	-0.23801726
2.28	0.45461369	0.22449826
2.3	0.44605539	0.61704999
2.32	0.43050295	0.92391236
2.34	0.40967016	1.1450853
2.36	0.3852708	1.2805687
2.38	0.35901868	1.3303625
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83728	490.96093
0.32	126.2491	467.94698
0.34	117.11581	445.47175
0.36	108.42665	423.53427
0.38	100.17086	402.13337
0.4	92.337742	381.26776
0.42	84.916593	360.93596
0.44	77.896756	341.13638
0.46	71.267601	321.86733
0.48	65.018537	303.12709
0.5	59.139004	284.91399
0.52	53.618473	267.22651
0.54	48.446448	250.06338
0.56	43.612448	233.42376
0.58	39.106009	217.30734
0.6	34.916664	201.71456
0.62	31.033928	186.6467
0.64	27.447281	172.10604
0.66	24.146149	158.09598
0.68	21.119875	144.62107
0.7	18.3577	131.68707
0.72	15.848741	119.30083
0.74	13.581964	107.47017
0.76	11.546173	96.203686
0.78	9.7299947	85.510418
0.8	8.1218736	75.399494
0.82	6.7100738	65.879703
0.84	5.4826911	56.959026
0.86	4.4276736	48.64417
0.88	3.5328519	40.94011
0.9	2.7859777	33.849698
0.92	2.1747696	27.373345
0.94	1.6869644	21.508819
0.96	1.3103704	16.251181
0.98	1.0329212	11.59286
1.	0.84272655	7.5238832
1.02	0.72811659	4.0322463
1.04	0.67767783	1.1044075
1.06	0.68027817	-1.2741236
1.08	0.72507941	-3.1181379
1.1	0.80153662	-4.4420597
1.12	0.89938429	-5.2593021
1.14	1.0086101	-5.5816865
1.16	1.1194182	-5.4189598
1.18	1.2221838	-4.7784376
1.2	1.3079961	-3.7814177
1.22	1.3729477	-2.7018763
1.24	1.4155914	-1.5502967
1.26	1.4344593	-0.32359983
1.28	1.4280005	0.98338336
1.3	1.3945433	2.3775069
1.32	1.3322656	3.8668551
1.34	1.240627	5.2397756
1.36	1.1249346	6.2737134
1.38	0.99187834	6.9776459
1.4	0.84797185	7.360096
1.42	0.69956588	7.4288054
1.44	0.55286599	7.1905225
1.46	0.41394539	6.656874
1.48	0.28721417	6.0120079
1.5	0.17362492	5.3431286
1.52	0.073634354	4.652417
1.54	-0.012334347	3.9410652
1.56	-0.083873087	3.2094149
1.58	-0.14057336	2.4571095
1.6	-0.18205101	1.6960746
1.62	-0.20908792	1.0263292
1.64	-0.22384732	0.46805061
1.66	-0.22854186	0.019557624
1.68	-0.22535006	-0.32086913
1.7	-0.21641631	-0.55491003
1.72	-0.2038523	-0.68414995
1.74	-0.18964141	-0.73234325
1.76	-0.17456612	-0.77543908
1.78	-0.15861015	-0.82058501
1.8	-0.14172256	-0.86874899
1.82	-0.12383462	-0.9207438
1.84	-0.10486277	-0.97724021
1.86	-0.084711344	-1.0387827
1.88	-0.063291384	-1.1030022
1.9	-0.040600458	-1.1658832
1.92	-0.016663525	-1.2276417
1.94	0.0084990943	-1.2884813
1.96	0.034870635	-1.348556
1.98	0.062437006	-1.4079806
2.	0.091186062	-1.4667946
2.02	0.1210707	-1.5202372
2.04	0.1519384	-1.565107
2.06	0.18361804	-1.6014353
2.08	0.21593902	-1.6292441
2.1	0.24873111	-1.6485486
2.12	0.28182434	-1.6593593
2.14	0.31504144	-1.6582796
2.16	0.34774304	-1.5999858
2.18	0.37856508	-1.4703233
2.2	0.4060802	-1.2692942
2.22	0.42886108	-0.9968997
2.24	0.44548043	-0.65314076
2.26	0.45451095	-0.2380179
2.28	0.45461368	0.22449789
2.3	0.44605538	0.61704977
2.32	0.43050295	0.92391224
2.34	0.40967016	1.1450852
2.36	0.3852708	1.2805686
2.38	0.35901868	1.3303625
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.8366	490.97253
0.32	126.24815	467.96278
0.34	117.11449	445.49306
0.36	108.42483	423.56271
0.38	100.16839	402.17098
0.4	92.334405	381.31698
0.42	84.912132	360.99974
0.44	77.890844	341.2182
0.46	71.25984	321.97123
0.48	65.008439	303.25768
0.5	59.125984	285.07644
0.52	53.601839	267.42649
0.54	48.425388	250.307
0.56	43.586027	233.71739
0.58	39.073161	217.65748
0.6	34.876195	202.12753
0.62	30.984521	187.12842
0.64	27.387509	172.66165
0.66	24.074491	158.7295
0.68	21.034745	145.33501
0.7	18.257481	132.482
0.72	15.731825	120.17504
0.74	13.446804	108.41936
0.76	11.391337	97.220672
0.78	9.5542239	86.585016
0.8	7.9241426	76.518472
0.82	6.4896518	67.0269
0.84	5.2391978	58.11562
0.86	4.1611283	49.789103
0.88	3.2437128	42.050667
0.9	2.4751681	34.902212
0.92	1.8436889	28.344013
0.94	1.337482	22.374582
0.96	0.94480205	16.990626
0.98	0.65398711	12.187098
1.	0.45349264	7.9573483
1.02	0.3319211	4.2933751
1.04	0.27804584	1.1861504
1.06	0.28082779	-1.3739861
1.08	0.32942383	-3.3968952
1.1	0.41318634	-4.8921929
1.12	0.52165399	-5.8688214
1.14	0.64453425	-6.3346616
1.16	0.77167888	-6.2962113
1.18	0.89305368	-5.7583477
1.2	0.99929842	-4.8408118
1.22	1.0860343	-3.8169261
1.24	1.1513382	-2.6973958
1.26	1.1932791	-1.4801682
1.28	1.2098709	-0.16179723
1.3	1.1990465	1.2622864
1.32	1.1586389	2.7974723
1.34	1.0878192	4.2291609
1.36	0.99166662	5.3317508
1.38	0.87670329	6.1112154
1.4	0.74933418	6.5732365
1.42	0.61585556	6.7229752
1.44	0.48246689	6.5649305
1.46	0.35527661	6.1088543
1.48	0.2387636	5.5374126
1.5	0.13397498	4.9367167
1.52	0.041480121	4.3082205
1.54	-0.038173935	3.6527189
1.56	-0.10445022	2.9704396
1.58	-0.15681149	2.2611447
1.6	-0.19474912	1.5370596
1.62	-0.2189279	0.89863311
1.64	-0.23140347	0.36655773
1.66	-0.23429175	-0.060287418
1.68	-0.22968586	-0.38304855
1.7	-0.21965622	-0.6028459
1.72	-0.20625139	-0.72073607
1.74	-0.19140182	-0.75998961
1.76	-0.17584618	-0.7961237
1.78	-0.15953251	-0.83590876
1.8	-0.14238116	-0.8799901
1.82	-0.12430063	-0.92890958
1.84	-0.10518952	-0.98311436
1.86	-0.084938382	-1.0429675
1.88	-0.063447709	-1.1059546
1.9	-0.04070712	-1.1679462
1.92	-0.016735642	-1.2290693
1.94	0.0084507743	-1.2894597
1.96	0.034838553	-1.3492202
1.98	0.062415897	-1.4084272
2.	0.0911723	-1.467092
2.02	0.12106181	-1.5204335
2.04	0.15193271	-1.5652353
2.06	0.18361443	-1.6015183
2.08	0.21593675	-1.6292973
2.1	0.2487297	-1.6485824
2.12	0.28182347	-1.6593806
2.14	0.31504091	-1.6582928
2.16	0.34774272	-1.599994
2.18	0.37856489	-1.4703283
2.2	0.40608008	-1.2692972
2.22	0.42886101	-0.99690152
2.24	0.44548039	-0.65314184
2.26	0.45451093	-0.23801854
2.28	0.45461366	0.22449751
2.3	0.44605538	0.61704956
2.32	0.43050294	0.92391212
2.34	0.40967015	1.1450851
2.36	0.3852708	1.2805686
2.38	0.35901868	1.3303625
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83592	490.98413
0.32	126.24719	467.97858
0.34	117.11317	445.51436
0.36	108.42301	423.59116
0.38	100.16592	402.20858
0.4	92.331068	381.3662
0.42	84.90767	361.06353
0.44	77.884933	341.30002
0.46	71.252078	322.07513
0.48	64.998341	303.38827
0.5	59.112964	285.23889
0.52	53.585205	267.62648
0.54	48.404328	250.55062
0.56	43.559605	234.01103
0.58	39.040312	218.00762
0.6	34.835725	202.54051
0.62	30.935114	187.61013
0.64	27.327737	173.21726
0.66	24.002833	159.36302
0.68	20.949615	146.04894
0.7	18.157261	133.27692
0.72	15.614909	121.04925
0.74	13.311645	109.36854
0.76	11.236502	98.237659
0.78	9.3784532	87.659614
0.8	7.7264115	77.63745
0.82	6.2692298	68.174096
0.84	4.9957045	59.272213
0.86	3.8945831	50.934036
0.88	2.9545738	43.161223
0.9	2.1643585	35.954727
0.92	1.5126082	29.314682
0.94	0.98799964	23.240346
0.96	0.57923373	17.730071
0.98	0.27505302	12.781335
1.	0.064258741	8.3908133
1.02	-0.064274377	4.5545039
1.04	-0.12158615	1.2678933
1.06	-0.11862259	-1.4738487
1.08	-0.066231757	-3.6756526
1.1	0.024836055	-5.3423262
1.12	0.14392368	-6.4783408
1.14	0.28045838	-7.0876367
1.16	0.42393953	-7.1734629
1.18	0.56392355	-6.7382579
1.2	0.69060078	-5.900206
1.22	0.79912091	-4.931976
1.24	0.88708499	-3.8444949
1.26	0.95209901	-2.6367366
1.28	0.99174124	-1.3069778
1.3	1.0035496	0.14706596
1.32	0.98501213	1.7280895
1.34	0.93501149	3.2185461
1.36	0.8583986	4.3897882
1.38	0.76152823	5.2447848
1.4	0.65069652	5.7863769
1.42	0.53214524	6.017145
1.44	0.4120678	5.9393385
1.46	0.29660783	5.5608345
1.48	0.19031303	5.0628172
1.5	0.094325031	4.5303047
1.52	0.0093258884	3.9640241
1.54	-0.064013522	3.3643726
1.56	-0.12502736	2.7314642
1.58	-0.17304963	2.06518
1.6	-0.20744723	1.3780447
1.62	-0.22876787	0.77093707
1.64	-0.23895963	0.26506485
1.66	-0.24004164	-0.14013246
1.68	-0.23402167	-0.44522796
1.7	-0.22289612	-0.65078177
1.72	-0.20865048	-0.75732219
1.74	-0.19316223	-0.78763598
1.76	-0.17712624	-0.81680832
1.78	-0.16045487	-0.85123252
1.8	-0.14303976	-0.89123122
1.82	-0.12476664	-0.93707536
1.84	-0.10551628	-0.98898851
1.86	-0.08516542	-1.0471522
1.88	-0.063604034	-1.108907
1.9	-0.040813782	-1.1700091
1.92	-0.01680776	-1.2304969
1.94	0.0084024543	-1.2904382
1.96	0.034806471	-1.3498845
1.98	0.062394789	-1.4088739
2.	0.091158538	-1.4673895
2.02	0.12105292	-1.5206297
2.04	0.15192702	-1.5653635
2.06	0.18361082	-1.6016013
2.08	0.21593448	-1.6293505
2.1	0.24872828	-1.6486162
2.12	0.28182259	-1.6594018
2.14	0.31504037	-1.6583061
2.16	0.3477424	-1.6000022
2.18	0.37856469	-1.4703333
2.2	0.40607997	-1.2693002
2.22	0.42886094	-0.99690334
2.24	0.44548035	-0.65314292
2.26	0.4545109	-0.23801918
2.28	0.45461365	0.22449714
2.3	0.44605537	0.61704934
2.32	0.43050294	0.92391199
2.34	0.40967015	1.1450851
2.36	0.3852708	1.2805686
2.38	0.35901867	1.3303624
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83524	490.99573
0.32	126.24624	467.99437
0.34	117.11184	445.53566
0.36	108.4212	423.6196
0.38	100.16344	402.24619
0.4	92.327731	381.41542
0.42	84.903208	361.12731
0.44	77.879021	341.38184
0.46	71.244317	322.17903
0.48	64.988242	303.51886
0.5	59.099945	285.40134
0.52	53.568571	267.82647
0.54	48.383268	250.79425
0.56	43.533184	234.30467
0.58	39.007464	218.35775
0.6	34.795256	202.95348
0.62	30.885707	188.09185
0.64	27.267964	173.77287
0.66	23.931174	159.99655
0.68	20.864485	146.76287
0.7	18.057042	134.07184
0.72	15.497993	121.92346
0.74	13.176486	110.31773
0.76	11.081667	99.254646
0.78	9.2026825	88.734212
0.8	7.5286805	78.756428
0.82	6.0488077	69.321293
0.84	4.7522111	60.428806
0.86	3.6280378	52.078969
0.88	2.6654347	44.27178
0.9	1.8535489	37.007241
0.92	1.1815274	30.28535
0.94	0.63851725	24.106109
0.96	0.21366541	18.469516
0.98	-0.10388106	13.375573
1.	-0.32497516	8.8242783
1.02	-0.46046986	4.8156328
1.04	-0.52121813	1.3496362
1.06	-0.51807297	-1.5737113
1.08	-0.46188734	-3.9544099
1.1	-0.36351423	-5.7924595
1.12	-0.23380662	-7.0878601
1.14	-0.083617485	-7.8406117
1.16	0.076200191	-8.0507144
1.18	0.23479343	-7.718168
1.2	0.38190314	-6.9596002
1.22	0.5122075	-6.0470258
1.24	0.62283179	-4.991594
1.26	0.71091888	-3.7933049
1.28	0.77361161	-2.4521584
1.3	0.80805283	-0.96815452
1.32	0.81138541	0.65870675
1.34	0.78220377	2.2079314
1.36	0.72513059	3.4478257
1.38	0.64635318	4.3783543
1.4	0.55205885	4.9995174
1.42	0.44843492	5.3113148
1.44	0.3416687	5.3137465
1.46	0.23793905	5.0128148
1.48	0.14186246	4.5882219
1.5	0.054675087	4.1238928
1.52	-0.022828344	3.6198276
1.54	-0.08985311	3.0760263
1.56	-0.14560449	2.4924889
1.58	-0.18928776	1.8692153
1.6	-0.22014534	1.2190297
1.62	-0.23860785	0.64324103
1.64	-0.24651578	0.16357197
1.66	-0.24579152	-0.2199775
1.68	-0.23835748	-0.50740737
1.7	-0.22613603	-0.69871764
1.72	-0.21104957	-0.79390832
1.74	-0.19492263	-0.81528235
1.76	-0.1784063	-0.83749295
1.78	-0.16137723	-0.86655628
1.8	-0.14369836	-0.90247234
1.82	-0.12523265	-0.94524113
1.84	-0.10584303	-0.99486266
1.86	-0.085392458	-1.0513369
1.88	-0.063760358	-1.1118594
1.9	-0.040920443	-1.172072
1.92	-0.016879878	-1.2319245
1.94	0.0083541344	-1.2914167
1.96	0.034774389	-1.3505487
1.98	0.062373681	-1.4093205
2.	0.091144775	-1.4676869
2.02	0.12104403	-1.5208259
2.04	0.15192132	-1.5654917
2.06	0.18360721	-1.6016843
2.08	0.21593221	-1.6294037
2.1	0.24872687	-1.64865
2.12	0.28182172	-1.6594231
2.14	0.31503984	-1.6583193
2.16	0.34774207	-1.6000104
2.18	0.3785645	-1.4703383
2.2	0.40607985	-1.2693033
2.22	0.42886088	-0.99690516
2.24	0.44548031	-0.65314401
2.26	0.45451088	-0.23801981
2.28	0.45461364	0.22449677
2.3	0.44605536	0.61704913
2.32	0.43050293	0.92391187
2.34	0.40967015	1.145085
2.36	0.3852708	1.2805685
2.38	0.35901867	1.3303624
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83456	491.00733
0.32	126.24529	468.01017
0.34	117.11052	445.55697
0.36	108.41938	423.64805
0.38	100.16097	402.28379
0.4	92.324394	381.46465
0.42	84.898746	361.19109
0.44	77.873109	341.46366
0.46	71.236555	322.28292
0.48	64.978144	303.64945
0.5	59.086925	285.56379
0.52	53.551937	268.02645
0.54	48.362209	251.03787
0.56	43.506762	234.59831
0.58	38.974615	218.70789
0.6	34.754787	203.36645
0.62	30.8363	188.57357
0.64	27.208192	174.32849
0.66	23.859516	160.63007
0.68	20.779355	147.4768
0.7	17.956823	134.86676
0.72	15.381078	122.79767
0.74	13.041327	111.26691
0.76	10.926831	100.27163
0.78	9.0269118	89.808811
0.8	7.3309495	79.875406
0.82	5.8283857	70.468489
0.84	4.5087178	61.5854
0.86	3.3614925	53.223902
0.88	2.3762957	45.382337
0.9	1.5427393	38.059755
0.92	0.85044669	31.256019
0.94	0.28903486	24.971872
0.96	-0.15190291	19.208961
0.98	-0.48281515	13.96981
1.	-0.71420906	9.2577434
1.02	-0.85666534	5.0767616
1.04	-0.92085012	1.4313791
1.06	-0.91752335	-1.6735739
1.08	-0.85754292	-4.2331673
1.1	-0.75186452	-6.2425928
1.12	-0.61153692	-7.6973795
1.14	-0.44769335	-8.5935868
1.16	-0.27153915	-8.9279659
1.18	-0.094336693	-8.6980782
1.2	0.073205509	-8.0189943
1.22	0.22529409	-7.1620756
1.24	0.3585786	-6.1386932
1.26	0.46973875	-4.9498733
1.28	0.55548197	-3.597339
1.3	0.61255602	-2.083375
1.32	0.63775868	-0.41067602
1.34	0.62939604	1.1973166
1.36	0.59186257	2.5058631
1.38	0.53117812	3.5119238
1.4	0.45342119	4.2126578
1.42	0.3647246	4.6054846
1.44	0.2712696	4.6881546
1.46	0.17927027	4.464795
1.48	0.093411891	4.1136265
1.5	0.015025143	3.7174809
1.52	-0.054982577	3.2756312
1.54	-0.1156927	2.78768
1.56	-0.16618162	2.2535135
1.58	-0.20552589	1.6732506
1.6	-0.23284345	1.0600147
1.62	-0.24844782	0.51554499
1.64	-0.25407193	0.062079085
1.66	-0.25154141	-0.29982254
1.68	-0.24269328	-0.56958678
1.7	-0.22937593	-0.74665352
1.72	-0.21344866	-0.83049444
1.74	-0.19668304	-0.84292871
1.76	-0.17968636	-0.85817757
1.78	-0.16229959	-0.88188003
1.8	-0.14435696	-0.91371346
1.82	-0.12569866	-0.95340691
1.84	-0.10616979	-1.0007368
1.86	-0.085619496	-1.0555216
1.88	-0.063916683	-1.1148118
1.9	-0.041027105	-1.174135
1.92	-0.016951995	-1.2333521
1.94	0.0083058144	-1.2923952
1.96	0.034742307	-1.3512129
1.98	0.062352573	-1.4097671
2.	0.091131013	-1.4679844
2.02	0.12103513	-1.5210221
2.04	0.15191563	-1.5656199
2.06	0.18360359	-1.6017673
2.08	0.21592994	-1.629457
2.1	0.24872545	-1.6486838
2.12	0.28182085	-1.6594443
2.14	0.3150393	-1.6583326
2.16	0.34774175	-1.6000185
2.18	0.3785643	-1.4703433
2.2	0.40607974	-1.2693063
2.22	0.42886081	-0.99690698
2.24	0.44548027	-0.65314509
2.26	0.45451086	-0.23802045
2.28	0.45461362	0.2244964
2.3	0.44605535	0.61704891
2.32	0.43050293	0.92391175
2.34	0.40967015	1.1450849
2.36	0.3852708	1.2805685
2.38	0.35901867	1.3303624
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83387	491.01893
0.32	126.24433	468.02597
0.34	117.1092	445.57827
0.36	108.41756	423.67649
0.38	100.1585	402.3214
0.4	92.321057	381.51387
0.42	84.894285	361.25487
0.44	77.867198	341.54548
0.46	71.228794	322.38682
0.48	64.968046	303.78003
0.5	59.073906	285.72624
0.52	53.535303	268.22644
0.54	48.341149	251.28149
0.56	43.480341	234.89195
0.58	38.941767	219.05802
0.6	34.714317	203.77942
0.62	30.786893	189.05529
0.64	27.148419	174.8841
0.66	23.787858	161.2636
0.68	20.694225	148.19073
0.7	17.856603	135.66168
0.72	15.264162	123.67188
0.74	12.906168	112.2161
0.76	10.771996	101.28862
0.78	8.851141	90.883409
0.8	7.1332184	80.994384
0.82	5.6079636	71.615686
0.84	4.2652245	62.741993
0.86	3.0949473	54.368835
0.88	2.0871566	46.492894
0.9	1.2319298	39.112269
0.92	0.51936595	32.226687
0.94	-0.060447536	25.837635
0.96	-0.51747123	19.948406
0.98	-0.86174924	14.564048
1.	-1.103443	9.6912084
1.02	-1.2528608	5.3378905
1.04	-1.3204821	1.513122
1.06	-1.3169737	-1.7734365
1.08	-1.2531985	-4.5119246
1.1	-1.1402148	-6.6927261
1.12	-0.98926722	-8.3068988
1.14	-0.81176922	-9.3465619
1.16	-0.61927849	-9.8052174
1.18	-0.42346682	-9.6779883
1.2	-0.23549213	-9.0783885
1.22	-0.061619316	-8.2771255
1.24	0.094325397	-7.2857923
1.26	0.22855861	-6.1064416
1.28	0.33735234	-4.7425196
1.3	0.4170592	-3.1985955
1.32	0.46413196	-1.4800588
1.34	0.47658831	0.18670185
1.36	0.45859455	1.5639005
1.38	0.41600307	2.6454933
1.4	0.35478352	3.4257983
1.42	0.28101428	3.8996544
1.44	0.2008705	4.0625626
1.46	0.12060149	3.9167753
1.48	0.044961321	3.6390312
1.5	-0.024624802	3.3110689
1.52	-0.087136809	2.9314347
1.54	-0.14153229	2.4993337
1.56	-0.18675876	2.0145382
1.58	-0.22176402	1.4772858
1.6	-0.24554156	0.90099975
1.62	-0.2582878	0.38784895
1.64	-0.26162809	-0.039413796
1.66	-0.2572913	-0.37966759
1.68	-0.24702909	-0.6317662
1.7	-0.23261584	-0.79458939
1.72	-0.21584775	-0.86708057
1.74	-0.19844344	-0.87057508
1.76	-0.18096642	-0.87886219
1.78	-0.16322195	-0.89720379
1.8	-0.14501556	-0.92495458
1.82	-0.12616467	-0.96157269
1.84	-0.10649654	-1.006611
1.86	-0.085846534	-1.0597064
1.88	-0.064073008	-1.1177642
1.9	-0.041133767	-1.1761979
1.92	-0.017024113	-1.2347797
1.94	0.0082574944	-1.2933737
1.96	0.034710224	-1.3518772
1.98	0.062331465	-1.4102138
2.	0.091117251	-1.4682818
2.02	0.12102624	-1.5212184
2.04	0.15190994	-1.5657481
2.06	0.18359998	-1.6018503
2.08	0.21592767	-1.6295102
2.1	0.24872404	-1.6487176
2.12	0.28181997	-1.6594656
2.14	0.31503877	-1.6583458
2.16	0.34774142	-1.6000267
2.18	0.37856411	-1.4703483
2.2	0.40607962	-1.2693093
2.22	0.42886074	-0.9969088
2.24	0.44548023	-0.65314617
2.26	0.45451084	-0.23802109
2.28	0.45461361	0.22449603
2.3	0.44605535	0.6170487
2.32	0.43050293	0.92391162
2.34	0.40967015	1.1450849
2.36	0.3852708	1.2805684
2.38	0.35901867	1.3303624
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297348
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83319	491.03053
0.32	126.24338	468.04177
0.34	117.10788	445.59957
0.36	108.41575	423.70493
0.38	100.15602	402.359
0.4	92.31772	381.56309
0.42	84.889823	361.31866
0.44	77.861286	341.62731
0.46	71.221032	322.49072
0.48	64.957948	303.91062
0.5	59.060886	285.88868
0.52	53.518669	268.42643
0.54	48.320089	251.52511
0.56	43.453919	235.18559
0.58	38.908918	219.40816
0.6	34.673848	204.19239
0.62	30.737486	189.537
0.64	27.088647	175.43971
0.66	23.7162	161.89712
0.68	20.609094	148.90466
0.7	17.756384	136.45661
0.72	15.147246	124.54609
0.74	12.771008	113.16529
0.76	10.617161	102.30561
0.78	8.6753703	91.958007
0.8	6.9354874	82.113362
0.82	5.3875416	72.762882
0.84	4.0217311	63.898586
0.86	2.828402	55.513768
0.88	1.7980176	47.603451
0.9	0.92112017	40.164784
0.92	0.18828521	33.197356
0.94	-0.40992993	26.703398
0.96	-0.88303955	20.687851
0.98	-1.2406833	15.158285
1.	-1.4926769	10.124673
1.02	-1.6490563	5.5990193
1.04	-1.7201141	1.594865
1.06	-1.7164241	-1.8732991
1.08	-1.6488541	-4.7906819
1.1	-1.5285651	-7.1428594
1.12	-1.3669975	-8.9164182
1.14	-1.1758451	-10.099537
1.16	-0.96701784	-10.682469
1.18	-0.75259694	-10.657898
1.2	-0.54418976	-10.137783
1.22	-0.34853272	-9.3921753
1.24	-0.1699278	-8.4328914
1.26	-0.012621518	-7.26301
1.28	0.1192227	-5.8877002
1.3	0.22156239	-4.3138159
1.32	0.29050524	-2.5494416
1.34	0.32378058	-0.8239129
1.36	0.32532653	0.62193791
1.38	0.30082801	1.7790628
1.4	0.25614586	2.6389387
1.42	0.19730396	3.1938242
1.44	0.1304714	3.4369706
1.46	0.061932713	3.3687555
1.48	-0.003489248	3.1644358
1.5	-0.064274746	2.904657
1.52	-0.11929104	2.5872383
1.54	-0.16737187	2.2109874
1.56	-0.20733589	1.7755628
1.58	-0.23800215	1.2813211
1.6	-0.25823967	0.74198478
1.62	-0.26812777	0.26015291
1.64	-0.26918424	-0.14090668
1.66	-0.26304119	-0.45951263
1.68	-0.2513649	-0.69394561
1.7	-0.23585574	-0.84252526
1.72	-0.21824684	-0.90366669
1.74	-0.20020385	-0.89822144
1.76	-0.18224648	-0.89954681
1.78	-0.16414431	-0.91252754
1.8	-0.14567417	-0.93619569
1.82	-0.12663068	-0.96973847
1.84	-0.1068233	-1.0124851
1.86	-0.086073572	-1.0638911
1.88	-0.064229333	-1.1207166
1.9	-0.041240429	-1.1782609
1.92	-0.017096231	-1.2362073
1.94	0.0082091745	-1.2943521
1.96	0.034678142	-1.3525414
1.98	0.062310356	-1.4106604
2.	0.091103488	-1.4685793
2.02	0.12101735	-1.5214146
2.04	0.15190425	-1.5658764
2.06	0.18359637	-1.6019333
2.08	0.2159254	-1.6295634
2.1	0.24872262	-1.6487514
2.12	0.2818191	-1.6594869
2.14	0.31503823	-1.6583591
2.16	0.3477411	-1.6000349
2.18	0.37856391	-1.4703533
2.2	0.4060795	-1.2693124
2.22	0.42886067	-0.99691061
2.24	0.44548019	-0.65314725
2.26	0.45451081	-0.23802172
2.28	0.4546136	0.22449566
2.3	0.44605534	0.61704848
2.32	0.43050292	0.9239115
2.34	0.40967014	1.1450848
2.36	0.3852708	1.2805684
2.38	0.35901867	1.3303623
2.4	0.33262757	1.2944667
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297347
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")

SI_TABLE_POTENTIALS.append(
"""
0.3	135.83251	491.04213
0.32	126.24243	468.05756
0.34	117.10656	445.62087
0.36	108.41393	423.73338
0.38	100.15355	402.39661
0.4	92.314383	381.61231
0.42	84.885361	361.38244
0.44	77.855375	341.70913
0.46	71.213271	322.59462
0.48	64.947849	304.04121
0.5	59.047867	286.05113
0.52	53.502035	268.62642
0.54	48.29903	251.76874
0.56	43.427497	235.47923
0.58	38.87607	219.75829
0.6	34.633379	204.60536
0.62	30.68808	190.01872
0.64	27.028875	175.99532
0.66	23.644542	162.53064
0.68	20.523964	149.6186
0.7	17.656164	137.25153
0.72	15.030331	125.4203
0.74	12.635849	114.11447
0.76	10.462325	103.32259
0.78	8.4995996	93.032606
0.8	6.7377563	83.23234
0.82	5.1671195	73.910079
0.84	3.7782378	65.055179
0.86	2.5618567	56.658701
0.88	1.5088785	48.714008
0.9	0.61031059	41.217298
0.92	-0.14279553	34.168025
0.94	-0.75941232	27.569162
0.96	-1.2486079	21.427296
0.98	-1.6196174	15.752523
1.	-1.8819108	10.558138
1.02	-2.0452518	5.8601481
1.04	-2.1197461	1.6766079
1.06	-2.1158745	-1.9731617
1.08	-2.0445097	-5.0694393
1.1	-1.9169154	-7.5929927
1.12	-1.7447278	-9.5259375
1.14	-1.539921	-10.852512
1.16	-1.3147572	-11.55972
1.18	-1.0817271	-11.637809
1.2	-0.8528874	-11.197177
1.22	-0.63544613	-10.507225
1.24	-0.434181	-9.5799905
1.26	-0.25380165	-8.4195784
1.28	-0.098906936	-7.0328808
1.3	0.026065569	-5.4290364
1.32	0.11687851	-3.6188243
1.34	0.17097285	-1.8345276
1.36	0.19205851	-0.32002467
1.38	0.18565296	0.91263223
1.4	0.15750819	1.8520792
1.42	0.11359364	2.487994
1.44	0.060072305	2.8113786
1.46	0.0032639335	2.8207358
1.48	-0.051939818	2.6898404
1.5	-0.10392469	2.4982451
1.52	-0.15144527	2.2430418
1.54	-0.19321146	1.9226411
1.56	-0.22791302	1.5365875
1.58	-0.25424028	1.0853564
1.6	-0.27093778	0.58296981
1.62	-0.27796775	0.13245687
1.64	-0.27674039	-0.24239956
1.66	-0.26879108	-0.53935767
1.68	-0.2557007	-0.75612502
1.7	-0.23909565	-0.89046113
1.72	-0.22064593	-0.94025282
1.74	-0.20196425	-0.92586781
1.76	-0.18352654	-0.92023143
1.78	-0.16506667	-0.9278513
1.8	-0.14633277	-0.94743681
1.82	-0.12709669	-0.97790424
1.84	-0.10715005	-1.0183593
1.86	-0.08630061	-1.0680758
1.88	-0.064385657	-1.123669
1.9	-0.04134709	-1.1803238
1.92	-0.017168348	-1.2376349
1.94	0.0081608545	-1.2953306
1.96	0.03464606	-1.3532057
1.98	0.062289248	-1.411107
2.	0.091089726	-1.4688768
2.02	0.12100846	-1.5216108
2.04	0.15189855	-1.5660046
2.06	0.18359276	-1.6020163
2.08	0.21592313	-1.6296166
2.1	0.24872121	-1.6487852
2.12	0.28181823	-1.6595081
2.14	0.3150377	-1.6583723
2.16	0.34774078	-1.6000431
2.18	0.37856372	-1.4703583
2.2	0.40607939	-1.2693154
2.22	0.4288606	-0.99691243
2.24	0.44548015	-0.65314833
2.26	0.45451079	-0.23802236
2.28	0.45461358	0.22449528
2.3	0.44605533	0.61704827
2.32	0.43050292	0.92391138
2.34	0.40967014	1.1450847
2.36	0.38527079	1.2805684
2.38	0.35901867	1.3303623
2.4	0.33262757	1.2944666
2.42	0.30766721	1.2007223
2.44	0.28455721	1.1114009
2.46	0.26316625	1.028818
2.48	0.24335956	0.95297347
2.5	0.22500239	0.88386739
2.52	0.20795995	0.82149972
2.54	0.19209748	0.76587048
2.56	0.17727835	0.71725847
2.58	0.16335849	0.6759427
2.6	0.150192	0.64192316
2.62	0.13763293	0.61519985
2.64	0.12553536	0.59577277
2.66	0.11375337	0.58364193
2.68	0.10214147	0.57851969
2.7	0.090622747	0.57264939
2.72	0.079263628	0.56255919
2.74	0.068148512	0.54824909
2.76	0.057361797	0.5297191
2.78	0.046987881	0.50696921
2.8	0.037111161	0.47999942
2.82	0.027819947	0.44749761
2.84	0.019319544	0.39965042
2.86	0.011949621	0.33444964
2.88	0.0060572492	0.25189526
2.9	0.001989501	0.15198729
2.92	0.000093448076	0.034725736
""")
