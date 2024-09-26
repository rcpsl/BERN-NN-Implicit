from tools_bounds_acasxu import Tools_Bounds
import torch
import sys
import json
import subprocess
from vnnlib import VNNLib_parser
from utils import get_acas_weights
import numpy as np
from utils import process_csv, cumulative_sum_sorted







class Verify_ACASXU():
    def __init__(self, net_pair, spec, timeout):
        self.net_pair = net_pair
        self.spec = spec
        self.timeout = timeout
        self.prev, self.tau = net_pair
        self.onnx_file = "/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/hscc23/bernn/HSCC_23_REP/experiments/acasxu/onnx/ACASXU_run2a_{prev}_{tau}_batch_2000.onnx".format(prev=self.prev, tau=self.tau)
        self.spec_file = "/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/hscc23/bernn/HSCC_23_REP/experiments/acasxu/vnnlib/prop_{spec}.vnnlib".format(spec=self.spec)
        self.input_space = VNNLib_parser(self.spec_file).read_vnnlib_simple(self.spec_file, 5, 5)[0][0]
        self.acasxu_spec = VNNLib_parser(self.spec_file).read_vnnlib_simple(self.spec_file, 5, 5)[0][1]





    def verify_acasxu_sia(self):
        ### sia_bounds : list of sia bounds
        ### returns : bool : True if sia verify spec, False otherwise

        # Serialize the parameters into command line arguments
        args = [sys.executable, 'tools_bounds_acasxu.py',
            '--tool', 'sia',
            '--prev', str(self.prev),
            '--tau', str(self.tau),
            '--spec', str(self.spec)]
        
        
        # Execute the script as a subprocess
        try:
            result = subprocess.run(args, check=True, text=True, timeout = self.timeout, stdout=subprocess.PIPE)
            sia_bounds, sia_time = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            return False  # Or handle timeout appropriately

        for i in range(self.numb_consts):
            spec_i_left_upper = 0.0
            spec_i_left = self.acasxu_spec[i][0][0]
            spec_i_right = self.acasxu_spec[i][1][0]
            for j in range(5):
                if spec_i_left[j] > 0:
                    spec_i_left_upper += spec_i_left[j] * sia_bounds[j][1]
                else:
                    spec_i_left_upper += spec_i_left[j] * sia_bounds[j][0]

            if spec_i_left_upper > spec_i_right:
                return False
            
        return True, sia_time
    
    def verify_acasxu_crown(self):
        ### crown_bounds : list of crown bounds
        ### returns : bool : True if crown verify spec, False otherwise
        
        # Serialize the parameters into command line arguments
        args = [sys.executable, 'tools_bounds_acasxu.py',
            '--tool', 'crown',
            '--prev', str(self.prev),
            '--tau', str(self.tau),
            '--spec', str(self.spec)]
        
        # Execute the script as a subprocess
        try:
            result = subprocess.run(args, check=True, text=True, timeout = self.timeout)
            with open('crown_bounds_time.txt', 'r') as f:
                data = json.load(f)
            crown_bounds = data['crown_bounds']
            crown_time = data['crown_time']
        except subprocess.TimeoutExpired:
            return False, 0

        #### and constraints ####
        if len(self.acasxu_spec) == 2:
            self.numb_consts = len(self.acasxu_spec[1])

            for i in range(self.numb_consts):
                spec_i_left_upper = 0.0
                spec_i_left = self.acasxu_spec[0][i, :]
                spec_i_right = self.acasxu_spec[1][i]
                for j in range(5):
                    if spec_i_left[j] > 0:
                        spec_i_left_upper += spec_i_left[j] * crown_bounds[j][1]
                    else:
                        spec_i_left_upper += spec_i_left[j] * crown_bounds[j][0]

                if spec_i_left_upper > spec_i_right:
                    return False, crown_time
            
            return True, crown_time
        
        #### or constraints ####
        else:
            self.numb_consts = len(self.acasxu_spec)

            for i in range(self.numb_consts):
                spec_i_left_upper = 0.0
                spec_i_left = self.acasxu_spec[i][0][0]
                spec_i_right = self.acasxu_spec[i][1][0]
                for j in range(5):
                    if spec_i_left[j] > 0:
                        spec_i_left_upper += spec_i_left[j] * crown_bounds[j][1]
                    else:
                        spec_i_left_upper += spec_i_left[j] * crown_bounds[j][0]

                if spec_i_left_upper <= spec_i_right:
                    return True, crown_time
            
            return False, crown_time
    


    

    def verify_acasxu_bern_nn_ibf(self, res):
        ### bern_ibf_bounds : list of bern ibf bounds
        ### returns : bool : True if bern ibf verify spec, False otherwise
        
        # Serialize the parameters into command line arguments
        args = [sys.executable, 'tools_bounds_acasxu.py',
            '--tool', 'bern_nn_ibf',
            '--prev', str(self.prev),
            '--tau', str(self.tau),
            '--spec', str(self.spec)]
        
        # Execute the script as a subprocess
        try:
            result = subprocess.run(args, check=True, text=True, timeout = self.timeout)
            with open('bern_ibf_bounds_time.txt', 'r') as f:
                data = json.load(f)
            bern_ibf_bounds = data['bern_ibf_bounds']
            bern_ibf_time = data['bern_ibf_time']
        except subprocess.TimeoutExpired:
            return False, 0  # Or handle timeout appropriately
        
        if res == 'sat':
            #### and constraints ####
            if len(self.acasxu_spec) == 2:
                self.numb_consts = len(self.acasxu_spec[1])

                for i in range(self.numb_consts):
                    spec_i_left_upper = 0.0
                    spec_i_left = self.acasxu_spec[0][i, :]
                    spec_i_right = self.acasxu_spec[1][i]
                    for j in range(5):
                        if spec_i_left[j] > 0:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][1]
                        else:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][0]

                    if spec_i_left_upper > spec_i_right:
                        return False, bern_ibf_time
                
                return True, bern_ibf_time
            
            #### or constraints ####
            else:
                self.numb_consts = len(self.acasxu_spec)

                for i in range(self.numb_consts):
                    spec_i_left_upper = 0.0
                    spec_i_left = self.acasxu_spec[i][0][0]
                    spec_i_right = self.acasxu_spec[i][1][0]
                    for j in range(5):
                        if spec_i_left[j] > 0:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][1]
                        else:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][0]

                    if spec_i_left_upper <= spec_i_right:
                        return True, bern_ibf_time
                
                return False, bern_ibf_time
            

        if res == 'unsat':
            #### and constraints ####
            if len(self.acasxu_spec) == 2:
                self.numb_consts = len(self.acasxu_spec[1])

                for i in range(self.numb_consts):
                    spec_i_left_upper = 0.0
                    spec_i_left = self.acasxu_spec[0][i, :]
                    spec_i_right = self.acasxu_spec[1][i]
                    for j in range(5):
                        if spec_i_left[j] > 0:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][0]
                        else:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][1]

                    if spec_i_left_upper > spec_i_right:
                        return True, bern_ibf_time
                
                return False, bern_ibf_time
            
            #### or constraints ####
            else:
                self.numb_consts = len(self.acasxu_spec)

                for i in range(self.numb_consts):
                    spec_i_left_upper = 0.0
                    spec_i_left = self.acasxu_spec[i][0][0]
                    spec_i_right = self.acasxu_spec[i][1][0]
                    for j in range(5):
                        if spec_i_left[j] > 0:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][0]
                        else:
                            spec_i_left_upper += spec_i_left[j] * bern_ibf_bounds[j][1]

                    if spec_i_left_upper <= spec_i_right:
                        return False, bern_ibf_time
                
                return True, bern_ibf_time
                
                

    



# spec_file = "/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/hscc23/bernn/HSCC_23_REP/experiments/acasxu/vnnlib/prop_7.vnnlib"
# vnnlib_parser = VNNLib_parser(spec_file)

# result = vnnlib_parser.read_vnnlib_simple(spec_file, 5, 5)
# print(result)
# print(result[0][1])




if __name__ == "__main__":


    # Process the provided CSV file again with the revised function
    csv_file_path = "/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/hscc23/bernn/HSCC_23_REP/experiments/acasxu/results.csv"
    actual_results = process_csv(csv_file_path)
    print(actual_results)  # Display a part of the dictionary to check the result

    instances = []

    for spec in range(1, 6):
        for a_prev in range(1, 6):
            for tau in range(1, 10):
                instances.append([a_prev, tau, spec])
    

    bern_ibf_times_instances = []
    crown_times_instances = []
    for instance in instances:
        if tuple(instance) in actual_results.keys():
            actual_res = actual_results[tuple(instance)]
            verify_acasxu_obj = Verify_ACASXU([instance[0], instance[1]], instance[2], 116)
            ### execute bern_ibf
            bern_ibf_result, bern_ibf_time  = verify_acasxu_obj.verify_acasxu_bern_nn_ibf(actual_res)
            print('##############################################################################################################################################################################')
            print('bern_ibf_result', bern_ibf_result)
            print('##############################################################################################################################################################################')
            if bern_ibf_result:
                bern_ibf_times_instances.append(bern_ibf_time)
            else:
                bern_ibf_times_instances.append(10000)

            ### execute crown
            crown_result, crown_time = verify_acasxu_obj.verify_acasxu_crown()
            if crown_result:
                crown_times_instances.append(crown_time)
            else:
                crown_times_instances.append(10000)
        else:
            continue


    print('bern_ibf_times_instances', bern_ibf_times_instances)
    print('crown_times_instances', crown_times_instances)


    # bern_ibf_times_instances = [2.1234467029571533, 2.116856813430786, 2.1574647426605225, 2.136155366897583, 2.1493353843688965, 2.3033034801483154, 2.211700916290283, 2.1718602180480957, 2.2186501026153564, 2.2079505920410156, 2.2293548583984375, 2.223644495010376, 2.1099438667297363, 2.2285263538360596, 2.254817485809326, 2.0789477825164795, 2.2771852016448975, 10000, 2.1179184913635254, 2.252229928970337, 2.1262340545654297, 2.1032795906066895, 2.191575765609741, 2.2568142414093018, 1.9859061241149902, 2.0499064922332764, 2.197047710418701, 2.1794424057006836, 2.2475955486297607, 2.18884539604187, 2.0786235332489014, 2.155574321746826, 2.2362191677093506, 10000, 2.218334436416626, 2.2570152282714844, 2.2588393688201904, 2.1023597717285156, 2.2027199268341064, 2.0979628562927246, 2.1571121215820312, 2.2476558685302734, 10000, 2.303283452987671, 2.1790895462036133, 2.0900301933288574, 2.14391827583313, 2.122744083404541, 2.237476348876953, 2.2892487049102783, 2.337749481201172, 2.292813301086426, 2.2037060260772705, 2.1928775310516357, 2.2039337158203125, 2.229154586791992, 2.2533462047576904, 2.126617193222046, 2.245196580886841, 2.2171669006347656, 10000, 10000, 10000, 2.168018102645874, 2.231391191482544, 2.141310453414917, 2.123065710067749, 2.1826250553131104, 10000, 1.9968209266662598, 2.0587222576141357, 2.274533271789551, 2.1861348152160645, 2.2474374771118164, 2.1636178493499756, 2.0535173416137695, 2.1596264839172363, 2.2253029346466064, 10000, 2.2257630825042725, 2.215411424636841, 2.1649019718170166, 2.0910897254943848, 2.186342239379883, 2.144244909286499, 2.1185131072998047, 2.279480457305908, 1.8909194469451904, 2.210693597793579, 2.1517744064331055, 2.001223087310791, 2.112077236175537, 2.1163830757141113, 2.1218018531799316, 2.0970325469970703, 2.100168228149414, 2.1391727924346924, 2.12781023979187, 2.0795769691467285, 2.118985652923584, 1.9004600048065186, 1.9411718845367432, 1.8924412727355957, 1.9826841354370117, 2.052896022796631, 1.9702587127685547, 2.0703940391540527, 2.012505054473877, 2.036895513534546, 1.9959988594055176, 1.952125072479248, 1.9463233947753906, 2.0472867488861084, 2.051412343978882, 2.0058982372283936, 2.1054911613464355, 1.9803993701934814, 2.002061605453491, 2.0574443340301514, 1.9686188697814941, 1.9120254516601562, 2.0010080337524414, 1.986151933670044, 1.9551734924316406, 2.0989668369293213, 1.9861280918121338, 1.9880452156066895, 2.008023738861084, 1.812375783920288, 1.9257471561431885, 1.995220422744751, 2.1063077449798584, 1.9527678489685059, 2.1729700565338135, 2.0386416912078857, 1.680375576019287]
    # crown_times_instances = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 7.126197099685669, 7.140364408493042, 7.528544187545776, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]


    # bern_ibf_times_instances = cumulative_sum_sorted(bern_ibf_times_instances)
    # crown_times_instances = cumulative_sum_sorted(crown_times_instances)

    # ### x_axis = np.arg([1, 2, ..., len(bern_ibf_times_instances)])
    # x_axis = np.arange(1, len(bern_ibf_times_instances) + 1)
    # ### plot bern_ibf_times_instances and crown_times_instances  versus x_axis and save the plot in the current directory
    # import matplotlib.pyplot as plt
    # plt.plot(x_axis, bern_ibf_times_instances, label = 'bern_ibf')
    # plt.plot(x_axis, crown_times_instances, label = 'crown')
    # plt.xlabel('number of verified instances')
    # plt.ylabel('time (s)')
    # plt.legend()
    # ### print yaxis in log scale
    # plt.yscale('log')
    # plt.savefig('bern_ibf_times_instances.png')
    






    