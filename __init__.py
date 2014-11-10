import copy
import re
import sys
import numpy as np

import itertools 

from itertools import combinations_with_replacement 
from numpy.linalg import inv as MatrixInverse

from matplotlib.mlab import find

__DESCRIPTION__="""A library to handle multi polinomial fitting."""

class Monomial(object) :
   def __init__(self,tupla) :
      self.t = tupla
   
   def __len__(self) :
      return len(self.t)

   def analyze(self,Nvars=None) :
      """ Returns the structure of a Monomial """
      a=np.array(self.t)
      
      if Nvars == None :
         nvars=a.max()+1
      else :
         nvars = Nvars

      powers=np.zeros(nvars,dtype=int)
      for i in range(nvars) :
         powers[i] = (a==i).sum()
      return {'powers':powers,'maxnvars':nvars,'deg':powers.sum(),'len':len(a),'nvars':len(np.unique(a))}
      
class MonomialBase(object) :
   __TYPE_MONOMIAL=type(Monomial((1,)))
   def __init__(self,nvars,maxdeg,mindeg=1,Fill=False) :
      self.nvars=nvars
      self.maxdeg = maxdeg
      self.mindeg=mindeg
      self.A=[]
      if Fill :
         self.A=self.Monomials()

      self.__current=0

   def banner(self) :
      return """
nvars = %d
mindeg= %d
maxdeg= %d
size  = %d""" % (self.nvars, self.mindeg, self.maxdeg,self.__len__())

   def __len__(self) :
      return len(self.A)

   def Monomials(self,nvars=None,mindeg=None,maxdeg=None) :
      if nvars==None :
         _nvars=self.nvars
      else :
         _nvars=nvars
      if maxdeg==None :
         _maxdeg=self.maxdeg
      else :
         _maxdeg=maxdeg
      if mindeg==None :
         _mindeg=self.mindeg
      else :
         _mindeg=mindeg

      List=[]
      items=[]
      for k in range(_nvars) :
         items.append(k)
      
      for deg in range(_mindeg,_maxdeg+1) :
         for uc in combinations_with_replacement(items, deg):
            List.append(uc)
      return List

   def __getitem__(self,i) :
      if i == type(slice(1)) :
         return self.A[i]
      if self.__len__() == 0 :
         return []
      if i < 0 :
         if abs(i) > self.__len__() :
            return []
         else :
            return self.A[i]
      if i > self.__len__() :
         return []
      return self.A[i]

   def __iter__(self) :
      return self

   def next(self) :
      if self.__current > self.__len__()-1 :
         self.__current = 0
         raise StopIteration
      else :
         self.__current+=1
         return self.A[self.__current-1]  

   def append(self,combination) :
      if combination==None :
         print "Invalid combination"
         return
      if len(combination) == 0 :
         print "Invalid combination"
         return
      if np.array(combination).min() < 0  :
         print "Invalid combination"
         return

      an=Monomial(combination).analyze(Nvars=self.nvars)

      if an['deg'] > self.maxdeg :
         self.maxdeg = an['deg']

      if an['deg'] < self.mindeg :
         self.mindeg = an['deg']

      if np.array(combination).max() > self.nvars :
         self.nvars = np.array(combination).max()

      self.A.append(combination)
      return

   def analyze(self) :
      if self.__len__() == 0 :
         return []
      l = {}
      u = Monomial((0,)).analyze(Nvars=self.nvars)
      for arg in u.keys() :
         if arg == 'powers' :
            l[arg] = np.zeros([self.__len__(),len(u[arg])],dtype='i')
         else :
            l[arg] = np.zeros([self.__len__()],dtype='i')
      irow=0
      for k in self.A :
         u = Monomial(k).analyze(Nvars=self.nvars)
         for arg in u.keys() :
            if arg == 'powers' :
               l[arg][irow,:]=np.array(u[arg][:])
            else :
               l[arg][irow]=u[arg]
         irow+=1
      return l

class NPolyFit :
   def __init__(self,sep=',',monomialBase=None) :
      self.sep=sep
      self.monomialBase=copy.deepcopy(monomialBase)

   def show(self) :
      if self.monomialBase != None :
         for k in self.monomialBase :
            print k
      else :
         print None
      
   def comb2key(self,iv) :
      l=[]
      for k in np.sort(iv) :
         l.append("%d"%k)
      if self.sep == None :
         return l
      return self.sep.join(l)

   def key2comb(self,key) :
      return np.array(key.split(self.sep),dtype=int)

   def combinations(self,nvars,maxdeg) :
      """deprecated"""
      List=[]
      items=[]
      for k in range(nvars) :
         items.append(k)
      
      for deg in range(1,maxdeg+1) :
         for uc in combinations_with_replacement(items, deg):
            List.append(uc)
      return List

   def joincombinations(self,comb1,comb2) :
      comb = []
      for k1 in comb1 :
         comb.append(k1)
      for k2 in comb2 :
         comb.append(k2)
      return np.sort(comb)

   def nvars2nunk(self,nvars,deg) :
      acc = []
      for i in range(deg+1) :
         if i == 0 :
            acc.append(0)
         elif i == 1:
            acc.append(nvars)
         elif i == 2:
            acc.append(nvars*(nvars+1)/2)
      return np.array(acc,dtype=int)

   def symbolicMatrix(self,nvars=None,maxdeg=None,size=2**8) :
      """symbolicMatrix"""
      if self.monomialBase == None :
         print "Generating monomial base"
         self.monomialBase = MonomialBase(nvars,maxdeg)
      _nvars=self.monomialBase.nvars
      _maxdeg=self.monomialBase.maxdeg
      nunk = len(self.monomialBase)
      M = np.zeros([nunk,nunk],dtype='S%d'%size)
      U = np.zeros(nunk,dtype='S%d'%size)
      V = np.zeros(nunk,dtype='S%d'%size)
      Sum = []
      
      SYSTEM=[]
      irow=0
      outmB = copy.deepcopy(self.monomialBase.A)
      inmB = copy.deepcopy(self.monomialBase.A)
      for kout in outmB :
         print irow,kout
         L=[]
         V[irow]=self.comb2key(kout)
         U[irow]='A'+self.comb2key(kout)
         icol=0
         for kin in inmB :
            Sidx = self.comb2key(self.joincombinations(kout,kin))
            
            Aidx = self.comb2key(kin)
            #print Sidx,Aidx
            L.append('S'+Sidx+'*'+'A'+Aidx)
            M[irow,icol] = Sidx
            icol+=1

            if len(find(np.array(Sum)==Sidx)) == 0 :
               Sum.append(Sidx)
         
         L="+".join(L)
         SYSTEM.append(L)
         irow+=1
      
      return {'M':M,'V':V,'S':Sum,'System':SYSTEM,'U':U,'nunk':nunk,'List':inmB}

   def npolyfit(self,X,D,maxdeg) :
      """ X is array X[vars,elements]: X[0,:] is X0 .... """
      nvars = len(X[:,0])
      ndat = len(X[0,:])
      
      program=symbolicMatrix(nvars,maxdeg)
      
      nunknow = program['nunk']
      if nunknow > ndat :
         print "Too much unknown with respect to the number of data "
         return None
      
      M = np.matrix([len(program['M'][0,:]),len(program['M'][:,0])])
      V = np.matrix(len(program['V']))
      S = {}
      
      # generates the sums
      irow=0
      for skey in program['S'] :
         acc=np.ones(ndat)
         comb=key2comb(skey)
         for c in comb :
            acc=acc*X[c,:]
         S[skey]=acc.sum()
         irow+=1
      
      # generates the Vsums
      irow=0
      for skey in program['V'] :
         acc=copy.deepcopy(D)
         comb=key2comb(skey)
         for c in comb :
            acc=acc*X[c,:]
         V[irow]=acc.sum()
         irow+=1
         
      # fills the matrix
      for irow in range(len(program['M'][0,:])) :
         for icol in range(len(program['M'][:,0])) :
            skey = program['M'][irow,icol]
            M[irow,icol]=1.*S[skey]
      
      # invert the matrix 
      try :
         IM=MatrixInverse(M)
         # performs the dotproduct
         P=IM*V
         
      except :
         print "Warning the matrix is not invertible"
         IM=None
         P=None
      
      return {'P':P,'M':M,'IM':IM,'S':S,'V':V,'U':program['U']}

if __name__ == '__main__':
   print "Testing ",sys.argv[0]

   nvars=3
   maxdeg=3
   
   print "\nnvars  %d \nmaxdeg %d \n" % (nvars,maxdeg)

   print "\nMonomials\n**************\n"

   MB = MonomialBase(nvars,maxdeg=2,Fill=True)
   print MB.banner()
   print

   MB.A=MB.Monomials(maxdeg=2)

   lst=MonomialBase(nvars,mindeg=3,maxdeg=3,Fill=True)
   dlst=lst.analyze()
   idx = np.where(dlst['nvars'] == 1)[0]
   for i in idx :
      MB.append(lst.A[i])

   for k in MB :
      print k

   print "\nPolyfit\n**************\n"
   MB.A=MB.Monomials(maxdeg=2)
   pF = NPolyFit(monomialBase=MB)
   print pF.monomialBase.banner()
   for k in pF.monomialBase :
      print k

   print pF.symbolicMatrix()

