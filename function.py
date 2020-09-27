from numpy import exp
############# PARA DEFINITION #############
'''
number of minima,
number of maxima,
number of saddle points
x label range
y label range
z label range
'''
############# EASY FUNCTIONS #############

######  Swedish Function ######
def s_f(x,y):
    return x**4 - 10*x**2 - x**3 + 6*y**2
def s_f_para():
    return 2,0,1,[-4,5],[-4,5],[-50,200]

###### Basket Function ######
def b_f(x,y):
    return (x**2 + y -11)**2 + (x + y**2 - 7)**2
def b_f_para():
    return 4,1,4,[-6,6],[-6,6],[0,2000]

###### Symmetric Basket Function ######
def sb_f(x,y):
    return x**2 + y**2 - 25 * exp(-((x-2)**2 + (y-2)**2))\
    - 25 * exp(-((x+2)**2 + (y+2)**2))\
    - 25 * exp(-((x-2)**2 + (y+2)**2))\
    - 25 * exp(-((x+2)**2 + (y-2)**2))\
    + 30 * exp(-(x**2 + y**2))
def sb_f_para():
    return 4,1,4,[-5,5],[-5,5],[-20,50]

###### Critical on Cross Function ######
def coc_f(x,y):
    return x**2 + y**2 - 8 * exp(-((x-2)**2 + (y-2)**2))\
    + 5 * exp(-(x**2 + y**2))\
    - 8 * exp(-((x+2)**2 + (y+2)**2))
def  coc_f_para():
    return 2,1,2,[-5,5],[-5,5],[-10,50]

###### Dumbbells Function ######
def d_f(x,y):
    return x**2 + y**2 - 14 * exp(-((x-2)**2 + (y-2)**2))\
    + 12 * exp(-((x-1)**2 + (y+1)**2))\
    - 14 * exp(-((x+2)**2 + (y+2)**2))\
    + 12 * exp(-((x+1)**2 + (y-1)**2))
def d_f_para():
    return 2,2,3,[-4,4],[-4,4],[-10,30]

###### Humble banana ######
def h_b(x,y):
    return x*exp(-x**2-y**2)+0.005*(exp(x)+exp(y)\
    +exp(-x)+exp(-y))
def h_b_para():
    return 1,1,1,[-3,3],[-3,3],[-0.5,0.5]

############ HARD FUNCTIONS ##########

###### Inverted banana mountains ######
def i_b(x,y):
    return 0.02*x**2 + 0.02*y**2 +0.05*x\
    +2*exp(-0.2*((x-3)**2+y**2))\
    -2.3*exp(-0.2*((x-1)**2+(y+3)**2))\
    -2.5*exp(-0.2*((x+5)**2+(y-3)**2))
def i_b_para():
    return 2,1,2,[-10,10],[-10,10],[-4,4]

###### Mountain pass ######
def m_p(x,y):
    return (y*exp(-x**2-y**2)+(x+2)*exp(-(x+2)**2-(y-2)**2))\
    +0.001*(exp(x)+exp(y)+exp(-x)+exp(-y))
def m_p_para():
    return 3,2,4,[-5,5],[-5,5],[-0.5,0.4]

###### Crazy mountain ######
def c_m(x,y):
    return 3*(1-x)**2*exp(-x**2-(y+1)**2)-10*(x/5-x**3-y**5)\
    *exp(-x**2-y**2)-1/3*exp(-(x+1)**2-y**2)\
    +0.01*(exp(-x) + exp(x) + exp(-y) + exp(y))
def c_m_para():
    return 4,3,6,[-6,6],[-6,6],[-10,8]

###### Dumbbells second function ######
def ds_f(x,y):
    return (x+0.8)**2+y**2-14*exp(-((x-2.2)**2+(y-0.4)**2))\
    +12*exp(-((x-1.9)**2+(y+1)**2))-14*exp(-((x+4.3)**2+(y+3.25)**2))\
    +12*exp(-((x+1.5)**2+(y-3)**2))
def ds_f_para():
    return 3,2,4,[-6,6],[-6,6],[-20,80]

###### Rocky function ######
def r_f(x,y):
    return x**2+y**2-20*exp(-((x-4.1)**2+(y-4.1)**2))-23.5*exp(-((x+4.5)**2+(y+4.5)**2))\
    -22.5*exp(-((x-4)**2+(y+4)**2))-20*exp(-((x+5)**2+(y-6)**2))+30*exp(-(x**2+y**2))\
    +15*exp(-((x+2)**2+(y+2)**2))+25*exp(-((x-4)**2+y**2))-10*exp(-((x+2)**2+(y-2)**2))\
    -13*exp(-((x+3)**2+(y-3)**2))-13*exp(-((x-2)**2+(y+2)**2))-13*exp(-((x-3.5)**2+(y+3.5)**2))\
    -10*exp(-((x-1.5)**2+(y-1.5)**2))-10*exp(-((x-2)**2+(y-2)**2))-15*exp(-(x**2+(y-4)**2))
def r_f_para():
    return 7,3,4,[-8,8],[-8,8],[-50,150]

###### Banana mountain ######
def b_m(x,y):
    return 0.02*x**2 + 0.02*y**2 + 0.05*x + 1-exp(-0.2*(x**2 + y**2 ))\
     + 1.7*exp(-0.2*((x-1)**2+(y+3)**2))+2*exp(-0.2*((x-5)**2+(y-3)**2))
def b_m_para():
    return 1,2,2,[-10,10],[-10,10],[0,5]

###### Marina trench ######
def m_t(x,y):
    return x**4-10*x**2-x**3 + 6*y**2-y**4 + 10*y**2 + y**3-6*x**2 + 0.5*exp(y) + 0.5*exp(-y)\
    -0.09*exp(x)-0.09*exp(-x)
def m_t_para():
    return 6,2,7,[-10,10],[-10,10],[-5000,8000]
