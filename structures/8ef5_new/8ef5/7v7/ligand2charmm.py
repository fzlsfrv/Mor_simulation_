"""ligand2charmm.py: This code produces CHARMM files using Antechamber."""
__author__ = "Kye Won Wang and Sang Jun Park"
__copyright__ = "Copyright 2021, Im Lab, Lehigh University"
__version__ = "3.0"
__note__ = "This code uses Amber20. If charmmgen is not available in Amber, please modify antechamber_last_flag and charmmgen running part below. \
            charge correction is applied to all the heavy atoms except hdrogens. \
            GAFF2 is used"
__email__ = "kyw220@lehigh.edu and sangjunpark@lehigh.edu"

import argparse
import os
import os.path
import subprocess
import re
from rdkit import Chem
from decimal import Decimal
#
#antechamber="/home/charmm-gui/local/miniconda3/bin/" 
#gaff="gaff2"
antechamber="/home/charmm-gui/local/miniconda3/bin/" 
os.environ["AMBERHOME"] = '/home/charmm-gui/local/miniconda3/'
chrg_factor=float(0.000001)
chrg_correction_max=10000; #
chrg_correction_tol = float(chrg_factor*chrg_correction_max); # the largest charge gap can be corrected
print ("The largest charge gap can be corrected by this code: ", chrg_correction_tol) #quit()
#
parser = argparse.ArgumentParser()
parser.add_argument('-Lname', dest='ligandname', required=True, help='Ligand name')
parser.add_argument('-nc', dest='netcharge', required=True, help='total net charge')
parser.add_argument('-index', dest='dumindex', required=True, help='index number')
parser.add_argument('-atomtype', dest='atomtype', default="gaff2", help='atom types, antechamber -L')
parser.add_argument('-charge_method', dest='charge_method', default="bcc", help='charge methods, antechamber -L')
args = parser.parse_args() #print(args.ligandname, args.netcharge)
#
input_ligand_name = args.ligandname
if args.netcharge == 'x':
  pass
else:
  input_netcharge = int(args.netcharge)
input_index = int(args.dumindex)
output_lc_name = args.ligandname.lower()
#
antechamber_last_flag="-fo ac -c {charge_method} -pf n -at {atomtype}".format(charge_method=args.charge_method, atomtype=args.atomtype) # If charmmgen is not available, use this command and activate "PLAN.B" part
#antechamber_last_flag="-fo prepc -c bcc -pf n" # If charmmgen is not available, use this command and activate "PLAN.B" part
#antechamber_last_flag="-fo charmm -c bcc -pf n" # This is for future purpose, when gaff2 is fully utilized (not beta)
# This is for chain indexing, A, B, C, ..., AA, AB, AC, ...
if input_index > 25:
  exceed = input_index - 25
  first = int((exceed - 1)/26)
  second = exceed - first * 26 - 1
  massappend = chr(first+ord('A'))+chr(second+ord('A'))
else:
  massappend = chr(ord('A')+input_index)
#
massappend='X'+massappend #print(massappend)
#
# This is for checking total charge from .mol2 file data
check_charge_mol2="cat "+input_ligand_name+".mol2 | awk \'{sum=sum+$9; if($1 == \"@<TRIPOS>BOND\"){exit}}END{print sum}\'"; #print(check_charge_mol2);

def read_mol2(filename):
    f = open(filename, 'r')
    read_line = False
    natom = 0
    for line in f:
        if line.startswith('#'): continue
        if not line.strip(): continue
        if line.startswith('@<TRIPOS>ATOM'):
            read_line = True
            continue
        elif line.startswith('@<TRIPOS>') and read_line:
            break
        if read_line:
            natom += 1
    f.close()
    return natom

#check_natom_mol2="cat "+input_ligand_name+".mol2 | head -3 | awk \'{print $1}\' | tail -1"
#check_natom_mol2=read_mol2(input_ligand_name+".mol2")

#
#natom_mol2=int(os.popen(check_natom_mol2).read()) #print(natom_mol2)
#natom_mol2=read_mol2(input_ligand_name+".mol2")
#charge_mol2=float(os.popen(check_charge_mol2).read()); #print((charge_mol2));
SDFfile = input_ligand_name+'.sdf'
suppl = Chem.SDMolSupplier(SDFfile)
if suppl[0] is None:
    suppl = Chem.SDMolSupplier(SDFfile, sanitize=False, removeHs=False, strictParsing=False)
for mol in suppl:
    charge_mol2 = Chem.GetFormalCharge(mol)
    print("charge: ", charge_mol2)
#
if charge_mol2 < 0.0:
  charge_mol2=int(charge_mol2-0.5)
else:
  charge_mol2=int(charge_mol2+0.5)
#
if args.netcharge == 'x':
   input_netcharge = charge_mol2
#
#if input_netcharge != charge_mol2:
#  print('Your input net charge(', input_netcharge, ') is not the same as total charge in .mol2 (', charge_mol2, '), please check')
#  quit ()
#else:
#  pass
#  
#antech_command=antechamber+" -rn "+args.ligandname+" -s 2 -nc "+str(input_netcharge)+" -i "+args.ligandname+".mol2 -fi mol2 -o "+output_lc_name+" "+antechamber_last_flag
antech_command1="OMP_NUM_THREADS=1 "+antechamber+"antechamber -rn "+args.ligandname+" -s 2 -nc "+str(input_netcharge)+" -i "+args.ligandname+".sdf -fi sdf -o "+output_lc_name+".ac "+antechamber_last_flag
print(antech_command1)
os.system(antech_command1)
antech_command2=antechamber+"parmchk2 -i "+output_lc_name+".ac -f ac -s 2 -o "+args.ligandname+".frcmod -a Y"
print(antech_command2)
os.system(antech_command2)
#
#######################################################################################################
# PLAN.B: If charmmgen is not available, use this bypass.   
#charmmgen="/share/ceph/woi216group-c2/shared/apps/amber16/bin/charmmgen" #charmmgen location  
#amberhome_for_charmmgen="/share/ceph/woi216group-c2/shared/apps/amber16" #charmmgen installed Amber
#os.environ["AMBERHOME"] = amberhome_for_charmmgen
#charmmgen_command=charmmgen+" -i ANTECHAMBER_PREP.AC -f ac -o "+output_lc_name+" -r "+args.ligandname
#######################################################################################################
# FOR SJ charmmgen_command=antechamber+"charmmgen -i "+output_lc_name+".ac -f ac -o "+output_lc_name+" -p "+args.ligandname+".frcmod -pf 2 -s 2"
charmmgen_command=antechamber+"charmmgen -i "+output_lc_name+".ac -f ac -o "+output_lc_name+" -p "+args.ligandname+".frcmod -pf 2 -s 2"
print(charmmgen_command)
os.system(charmmgen_command)
#
#check output
check_file = output_lc_name + ".rtf"
if not os.path.exists(check_file):
  print("file not exist, please check.")
  quit()
#
# create 5-decimal-charge rtf file
creat_5deci_command="cat "+check_file+" | awk \'{if ($1 == \"ATOM\"){cut=length($0)-1; dum=substr($0,1,cut); dum=dum\"\"0; print dum;}else{print $0}    }\' > "+check_file+".5deci"
#print(creat_5deci_command)
os.system(creat_5deci_command)
#
#charge correction
calc_charge_command="cat "+check_file+".5deci | grep ATOM | awk \'{split($4,a,\"\"); a[8]=0; dum=\"\"; for(i=1;i<=8;i++){dum=dum\"\"a[i];}; $4=dum; print $0 }\' | awk \'{sum=sum+$4;}END{print sum}\'"; #print(check_charge_mol2);
fnc = float(os.popen(calc_charge_command).read()) 
#print(input_netcharge, fnc) 
#quit()
os.system("mv "+output_lc_name+".rtf "+output_lc_name+".rtf.original")
os.system("mv "+output_lc_name+".rtf.5deci "+output_lc_name+".rtf.old")
os.system("mv "+output_lc_name+".prm "+output_lc_name+".prm.old")
os.system("head -n 3 "+output_lc_name+".prm.old > "+output_lc_name+".prm")
#
charge_gap = float("{0:.6f}".format(input_netcharge - fnc))
if charge_gap > 0.0:
  sign = 1.0
else:
  sign = -1.0
#
num_target=0
charge_target=[] 
#  
rtf_old = open(output_lc_name+".rtf.old",'r')
# 
while True:
  line = rtf_old.readline()
  if not line: break
  line=line.strip()
  if line.startswith ('ATOM'):
    line2=line.split()
    if line2[1][0] != 'H': 
      charge_target.append(float(line2[3]))
      num_target += 1
rtf_old.close() #print(charge_target) #quit()
#
def adjustCharge(parmed_obj):
    charge = 0.0
    for chg in parmed_obj:
        _charge = round(chg, 5)
        charge += _charge

    i = 0
    d_chrg = charge - round(charge)
    sign   = -1.0 if d_chrg > 0.0 else 1.0
    while round(d_chrg, 5) != 0.0:
        i = i % len(parmed_obj)
        parmed_obj[i] += sign*0.00001
        d_chrg += sign*0.00001
        i += 1
    return parmed_obj

charge_target = adjustCharge(charge_target)

#i=0
#n_times_correction=0;
#while charge_gap != 0.0:
#  temp = float(charge_target[i]) + float(sign*chrg_factor)
#  charge_target[i] = float("{0:.5f}".format(temp)) 
#  charge_gap = float("{0:.5f}".format(float(charge_gap) - float(sign*chrg_factor)))
#  i+=1
#  n_times_correction += 1
#  if i == num_target:
#    i=0
#  if n_times_correction > chrg_correction_max:
#    print("the number of charge correction was over the max number of correction allowed, please check\n")
#    break 
rtf_old = open(output_lc_name+".rtf.old",'r')
rtf_new = open(output_lc_name+".rtf",'w')
prm = open(output_lc_name+".prm", 'a')
prm.write("ATOMS\n")
#
num_heavy = 0
new_charge = 0.0
xxx = []
debug = 0.0 
debug2 = 0.0
rtf_lines = []
pointer = None
while True:
  line = rtf_old.readline()
  if not line: break
#
  line2 = line.strip()
  line2 = line2.split()
#
  if line2:
    if line2[0] == 'MASS':
      newAtomType = "_"+line2[2].upper()+massappend
#      line=line.replace(line2[2]+"   ", newAtomType)
      line2[2] = newAtomType

      if len(line2) == 4:
        rtf_lines.append("MASS   %3d %-6s %9.6f\n" % (-1, line2[2], float(line2[3])))
        prm.write("MASS   %3d %-6s %9.6f\n" % (-1, line2[2], float(line2[3]))) 
      else:
        addition = ""
        for x in range(4,len(line2)):
          addition = addition+" "+line2[x]
          rtf_lines.append("MASS   %3d %-6s %9.6f  %s\n" % (-1, line2[2], float(line2[3]), addition))
          prm.write("MASS   %3d %-6s %9.6f  %s\n" % (-1, line2[2], float(line2[3]), addition)) 

    elif line2 [0] == 'ATOM':
      newAtomType = "_"+line2[2].upper()+massappend
      line2[2] = newAtomType
      charge = float(line2[3])
      element = str(line2[1])
      element = element[0]

      if element != 'H':
        rtf_lines.append("%s %-5s %-6s %9.6f\n" % (line2[0], line2[1], line2[2], charge_target[num_heavy]))
        if not pointer: pointer = "%s %-5s %-6s %9.6f\n" % (line2[0], line2[1], line2[2], charge_target[num_heavy])
        debug += charge_target[num_heavy]
        debug2 += float("%9.6f" % charge_target[num_heavy])
        xxx.append("%9.6f" % charge_target[num_heavy])
        new_charge = new_charge + charge_target[num_heavy]
        num_heavy += 1
      else:
        #rtf_new.write("%s %-5s %-6s %9.6f\n" % (line2[0], line2[1], line2[2], charge))
        rtf_lines.append("%s %-5s %-6s %9.6f\n" % (line2[0], line2[1], line2[2], charge))
        debug += charge
        debug2 += float("%9.6f" % charge)
        new_charge = new_charge + charge    
        xxx.append("%9.6f" % charge)
        
    elif line2 [0] == 'ANGL':
      pass
    elif line2 [0] == 'DIHE':
      pass
    else:
      #rtf_new.write(line)
      rtf_lines.append(line)
  else:
    line = "\n"
    #rtf_new.write(line)
    rtf_lines.append(line)
#
if not debug.is_integer():
    remainder = round(debug) - debug
    index = rtf_lines.index(pointer)
    head, sp1, sp2, charge = rtf_lines[index].split()
    test = round(remainder,6)
    rtf_lines[index] = "%s %-5s %-6s %9.6f\n" % (head, sp1, sp2, float(charge)+test)
rtf_new.writelines(rtf_lines)
rtf_old.close()
rtf_new.close()
if num_target != num_heavy:
  print ("Somthing wrong on charge correction, please check. Tot_heavy_atoms: %d, actually corrected atoms: %d \n" % (num_target, num))
#
CHG_TOTAL = Decimal(0)
for line_atom in rtf_lines:
    if line_atom.startswith('ATOM'):
        charge = Decimal(line_atom.split()[-1])
        CHG_TOTAL += charge
tmp=os.popen("cat "+output_lc_name+".rtf | grep RESI | awk '{print $0}'").read().strip()
tmp2=tmp.split()
old=tmp
new_charge="{:.3f}".format(new_charge)
print("new charge: ", new_charge)
print("CHG_TOTAL: ", CHG_TOTAL)
#tmp=tmp.replace(tmp2[2], str(new_charge))
tmp=tmp.replace(tmp2[2], str(CHG_TOTAL))
os.system("sed -i \"s/"+old+"/"+tmp+"/g\" "+output_lc_name+".rtf")
  
command="sed -i \"s/"+tmp2[1]+"/"+input_ligand_name+"/g\" "+output_lc_name+".rtf"
os.system(command)
print("Charges are corrected") 
#
# Post processing parameter file
#
prm_old = open(output_lc_name+".prm.old", 'r')
state=""
while True:
  line = prm_old.readline()
  if not line: break

  line2 = line.strip()
  if line2 == 'BOND':
    state = 'BOND'
  if line2 == 'ANGLE':
    state = 'ANGLE'
  if line2 == 'DIHEDRAL':
    state = 'DIHEDRAL'
  if line2 == 'IMPHI':
    state = 'IMPHI'
#    
  line3 = line2.split()
#  
  if line.startswith('NONBONDED'):
    state = 'NONBONDED'
    prm.write ("\n\nNONBONDED nbxmod  5 atom cdiel shift vatom vdistance vswitch -\n")
  elif line.startswith('CUTNB'):
    prm.write ("cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5\n")
  elif not state:
    pass 

  elif state == 'BOND':
    if re.match ("^(\S+)\s+(\S+)(.+)", line2): # SJ Check please
      line3[0]="_"+line3[0].upper()+massappend
      line3[1]="_"+line3[1].upper()+massappend
      prm.write ("%-3s %s%8.2f%8.3f\n" % (line3[0], line3[1], float(line3[2]), float(line3[3])))
    elif line2 == 'BOND':
      prm.write ("\nBONDS\n")
    else:
      prm.write (line2) 

  elif state == 'ANGLE':
    if re.match ("^(\S+)\s+(\S+)\s+(\S+)(.+)", line2):
      line3[0]="_"+line3[0].upper()+massappend
      line3[1]="_"+line3[1].upper()+massappend
      line3[2]="_"+line3[2].upper()+massappend
      if len(line3) == 5:
        prm.write ("%-3s %s %s%9.3f%12.3f\n" % (line3[0], line3[1], line3[2], float(line3[3]), float(line3[4])))
      else:
        addition = ""
        for x in range(5,len(line3)):
          addition = addition+" "+line3[x]
        prm.write ("%-3s %s %s%9.3f%12.3f  %s\n" % (line3[0], line3[1], line3[2], float(line3[3]), float(line3[4]), addition))
    elif line2 == 'ANGLE':
      prm.write ("\nANGLES\n")
    else:
      prm.write (line2)        

  elif state == 'DIHEDRAL':
    if re.match ("^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.+)", line2):
      for x in range (0, 4):
        if line3[x] == 'X': 
          pass
        else:
          line3[x]="_"+line3[x].upper()+massappend      
      if len(line3) == 7:
        prm.write ("%-6s%-6s%-6s%-6s%9.3f%10d%10.1f\n" % (line3[0], line3[1], line3[2], line3[3], float(line3[4]), int(line3[5]), float(line3[6])))
      else:
        addition = ""
        for x in range(7,len(line3)):
          addition = addition+" "+line3[x]
        prm.write ("%-6s%-6s%-6s%-6s%9.3f%10d%10.1f   %s\n" % (line3[0], line3[1], line3[2], line3[3], float(line3[4]), int(line3[5]), float(line3[6]), addition))
    elif line2 == 'DIHEDRAL':
      prm.write ("\nDIHEDRALS\n")
    else:
      prm.write (line2)        

  elif state == 'IMPHI':
    if re.match ("^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.+)", line2):
      for x in range (0, 4):
        if line3[x] == 'X': 
          pass
        else:
          line3[x]="_"+line3[x].upper()+massappend      
      if len(line3) == 7:
        prm.write ("%-6s%-6s%-6s%-6s%9.3f%10d%10.1f\n" % (line3[0], line3[1], line3[2], line3[3], float(line3[4]), int(line3[5]), float(line3[6])))
      else:
        addition = ""
        for x in range(7,len(line3)):
          addition = addition+" "+line3[x]
        prm.write ("%-6s%-6s%-6s%-6s%9.3f%10d%10.1f   %s\n" % (line3[0], line3[1], line3[2], line3[3], float(line3[4]), int(line3[5]), float(line3[6]), addition))
    elif line2 == 'IMPHI':
      prm.write ("\nIMPROPER\n")
    else:
      prm.write (line2)        

  elif state == 'NONBONDED':
    if re.match ("^(\w+)(.*)", line2):
      if len(line3) == 7:
        line3[0]="_"+line3[0].upper()+massappend
        prm.write ("%-6s%9.2f%10.4f%10.4f%10.2f%10.4f%10.4f\n" % (line3[0], float(line3[1]), float(line3[2]), float(line3[3]), float(line3[4]), float(line3[5]), float(line3[6])))
      else:
        addition = ""
        for x in range(7,len(line3)):
          addition = addition+" "+line3[x]
        prm.write ("%-6s%9.2f%10.4f%10.4f%10.2f%10.4f%10.4f   %s\n" % (line3[0], float(line3[1]), float(line3[2]), float(line3[3]), float(line3[4]), float(line3[5]), float(line3[6]), addition))
    elif re.match ("^\s+(\w+)(.*)", line2):
      if len(line3) == 7:
        line3[0]=" _"+line3[0].upper()+massappend
        ppm.write ("%-6s%9.2f%10.4f%10.4f%10.2f%10.4f%10.4f\n" % (line3[0], float(line3[1]), float(line3[2]), float(line3[3]), float(line3[4]), float(line3[5]), float(line3[6])))
      else:
        addition = ""
        for x in range(7,len(line3)):
          addition = addition+" "+line3[x]
        prm.write ("%-6s%9.2f%10.4f%10.4f%10.2f%10.4f%10.4f   %s\n" % (line3[0], float(line3[1]), float(line3[2]), float(line3[3]), float(line3[4]), float(line3[5]), float(line3[6]), addition))
    else:
      if line3[0] == '!':
        prm.write (line2+"\n")        
      else:
        prm.write (" "+line2+"\n")

  else:
    pass
prm.close()    
prm_old.close()  
#  
inp = open (output_lc_name+".inp",'r')  
crd = open (output_lc_name+".crd",'w')  
crd.write ("* Residues coordinate\n*\n")
#  
count1 = 0  
count2 = 0  
num = 0
while True:
  line = inp.readline()
  if not line: 
    if num !=count2:
      print ("number of atoms in .inp file has problems\n", num, count2)
      break  
    break
  line2 = line.strip()
  
  if re.match ("^\s*(\d+)$", line2):
    count1 += 1
    if count1 == 2:
      num = int(line2)
      crd.write ("%5d\n" % num)
  if re.match ("^\s*\S+\s+\S+\s+"+"MOL"+"\s+\S+\s+\S+\s+\S+\s+\S+$", line2):
    line3 = line2.split() 
    crd.write ("%5d%5d %-4s %-4s %9.5f %9.5f %9.5f\n" % (int(line3[0]), int(line3[1]), input_ligand_name, line3[3], float(line3[4]), float(line3[5]), float(line3[6]) ))
    count2 += 1
     
inp.close()
crd.close()
#  
#os.system("rm -f mopac.out mopac.pdb mopac.in divcon.pdb "+output_lc_name+".rtf.old "+output_lc_name+".prm.old")
#os.system("rm -f ANTECHAMBER* sqm* PREP.INF NEWPDB.PDB ATOMTYPE.INF "+output_lc_name+".rtf.old "+output_lc_name+".prm.old "+output_lc_name)

