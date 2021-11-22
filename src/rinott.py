import math 



# source https://people.orie.cornell.edu/cn254/MPIRnS_docs/_rinott_func_8h_source.html
def ZCDF(x):
    '''
    Parameters
        x	

    Returns
        The cumulative normal distribution function (CNDF) for a standard normal: N(0,1) evaluated at x. 
    '''
    if x < 0.0:
        neg = 1
    else:
        neg = 0
    
    if neg == 1:
        k = (1.0 / ( 1.0 + 0.2316419 * x))
        y = (((( 1.330274429 * k - 1.821255978) * \
            k + 1.781477937) * k - 0.356563782) *\
            k + 0.319381530) * k
    
    y = y = 1.0 - 0.3989422803   * math.exp(-0.5 * x * x) * y

    return (1 - neg) * y + neg * (1.0 - y)


def CHIPDF(N, C, lngam):
    '''
    Parameters
        N	Degree of freedom
        C	
        lngam	LNGAM(N) is LN(GAMMA(N/2))

    Returns
        The PDF of the Chi^2 distribution with N degress of freedom for N <= 50, evaluated at C 
    '''
    FLN2 = N/2
    TMP = -FLN2 * math.log(2) - lngam[N-1] + (FLN2 - 1.0) * math.log(C) - C/2
    return math.exp(TMP)

def rinott(T, PSTAR, NU):
    '''
    Parameters
        T	Number of systems
        PSTAR	1 - alpha, i.e. Power of the test. Usually set to 0.95
        NU	Stage-0 sample size - 1

    Returns
        The Rinott Constant 
    '''
    LNGAM = [None] * 50
    NU = min(NU, 50)
    WEX = [None] * 50
    W = [.10921834195238497114,
        .21044310793881323294,
        .23521322966984800539,
        .19590333597288104341,
        .12998378628607176061,
        .70578623865717441560E-1,
        .31760912509175070306E-1,
        .11918214834838557057E-1,
        .37388162946115247897E-2,
        .98080330661495513223E-3,
        .21486491880136418802E-3,
        .39203419679879472043E-4,
        .59345416128686328784E-5,
        .74164045786675522191E-6,
        .76045678791207814811E-7,
        .63506022266258067424E-8,
        .42813829710409288788E-9,
        .23058994918913360793E-10,
        .97993792887270940633E-12,
        .32378016577292664623E-13,
        .81718234434207194332E-15,
        .15421338333938233722E-16,
        .21197922901636186120E-18,
        .20544296737880454267E-20,
        .13469825866373951558E-22,
        .56612941303973593711E-25,
        .14185605454630369059E-27,
        .19133754944542243094E-30,
        .11922487600982223565E-33,
        .26715112192401369860E-37,
        .13386169421062562827E-41,
        .45105361938989742322E-47]

    X = [.44489365833267018419E-1,
        .23452610951961853745,
        .57688462930188642649,
        .10724487538178176330E1,
        .17224087764446454411E1,
        .25283367064257948811E1,
        .34922132730219944896E1,
        .46164567697497673878E1,
        .59039585041742439466E1,
        .73581267331862411132E1,
        .89829409242125961034E1,
        .10783018632539972068E2,
        .12763697986742725115E2,
        .14931139755522557320E2,
        .17292454336715314789E2,
        .19855860940336054740E2,
        .22630889013196774489E2,
        .25628636022459247767E2,
        .28862101816323474744E2,
        .32346629153964737003E2,
        .36100494805751973804E2,
        .40145719771539441536E2,
        .44509207995754937976E2,
        .49224394987308639177E2,
        .54333721333396907333E2,
        .59892509162134018196E2,
        .65975377287935052797E2,
        .72687628090662708639E2,
        .80187446977913523067E2,
        .88735340417892398689E2,
        .98829542868283972559E2,
        .11175139809793769521E3]

    for i in range(1, len(WEX) + 1):
        WEX[i - 1] = W[i - 1] * math.exp(X[i - 1])
    
    LNGAM[0] = 0.5723649429
    LNGAM[1] = 0.0
    
    for i in range(2, 26):
        LNGAM[2*i - 2] = math.log(i - 1.5) + LNGAM[2*i - 4]
        LNGAM[2*i - 1] = math.log(i - 1.0) + LNGAM[2*i - 3]

    DUMMY = 1
    H = 4
    LOWERH = 0
    UPPERH = 20
    ANS = 0
    TMP = 0
    for i in range(1, 33):
        ANS = 0
        for j in range (1, 33):
            TMP = 0
            for k in range(1, 33):
                TMP += WEX[k-1] * \
                    ZCDF( H / math.sqrt(NU * (1.0 / X[k-1]+1.0/X[j-1] ) / DUMMY )) * \
                    CHIPDF( NU, DUMMY * X[k-1], LNGAM ) * DUMMY
            TMP = math.pow(TMP, T - 1)      # using math.pow to ensure the result is floating point
            ANS = ANS + WEX[j-1] * TMP * CHIPDF(NU, DUMMY * X[j-1], LNGAM) * DUMMY
        if abs(ANS - PSTAR) <= 0.000001:
            return H
        elif ANS > PSTAR:
            UPPERH = H
            H = (LOWERH + UPPERH)/2.0
        else:
            LOWERH = H
            H = (LOWERH + UPPERH)/2.0
    
    return H

