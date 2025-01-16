import re
import json
import sys
import time
import math
import os
import io
import pandas as pd
import importlib
import threading
from typing import List, Any
from collections import Counter
import random
import itertools
import argparse

import sglang as sgl
from sglang import function, set_default_backend, RuntimeEndpoint
from sglang.lang.interpreter import ProgramState
from sglang.lang.backend.runtime_endpoint import SglSamplingParams
from sglang import function, system, user, assistant
from utils import extract_answer, math_equal, majority_voting
import numpy as np



import math
from collections import Counter, defaultdict
from typing import List


def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist:
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0

def norm(Olist):
    s = sum(Olist)
    return [o / s for o in Olist]

def count(Olist):
    x_dict = defaultdict(lambda: 0.0)
    for x in Olist:
        x_dict[x] += 1
    cc = [c for _,c in x_dict.items()]
    #print(cc)
    return cc

def item_entropy(answers: List) -> float:
    return entropy(norm(count(answers)))


def length_normalized_entropy(log_probs: List[float]) -> float:
    entropy = -sum(log_probs)
    return entropy / len(log_probs)

def even_distribute_resources(M: int, N: int) -> list[int]:
    if N <= 0 or M < 0:
        assert False, "Can not assign"
        
    base = M // N  # 
    extra = M % N  # extra resources
    
    distribution = []
    for i in range(N):
        if i < extra:
            distribution.append(base + 1)
        else:
            distribution.append(base)
            
    return distribution

def smart_distribute_resources(M: int, N: int, L: List, m: int) -> List[int]:
    assert len(L) == N and M >= N * m and all(x > 0 for x in L)
    
    remaining_resources = M - N * m    
    weights = np.array(L) / sum(L)    
    extra_distribution = np.floor(weights * remaining_resources).astype(int)    
    remainder = remaining_resources - sum(extra_distribution)
    if remainder > 0:
        indices = np.argsort(weights)[-int(remainder):]
        for idx in indices:
            extra_distribution[idx] += 1    
    final_distribution = [m + extra for extra in extra_distribution]
    
    # 
    assert sum(final_distribution) == M
    assert all(x >= m for x in final_distribution)
    
    return final_distribution

def distribute_parallel_size_with_limits(parallel_size, L):
    n = len(L)
    # Initialize the distribution list
    distribution = [0] * n
    
    # Calculate the base size each person should get
    base_size = parallel_size // n
    
    # Distribute the base size while respecting the limits
    for i in range(n):
        distribution[i] = min(base_size, L[i])
        parallel_size -= distribution[i]
    
    # Distribute the remaining parallel_size
    for i in range(n):
        if parallel_size == 0:
            break
        # Calculate the additional allocation possible
        additional_allocation = min(parallel_size, L[i] - distribution[i])
        distribution[i] += additional_allocation
        parallel_size -= additional_allocation
    
    return distribution
def find_max_x(L, M):
    # Initialize the binary search bounds
    left, right = 0, max(L)

    while left < right:
        mid = (left + right + 1) // 2
        # Calculate the sum of the list with all elements greater than mid replaced by mid
        current_sum = sum(min(x, mid) for x in L)
        
        if current_sum < M:
            left = mid  # mid is a valid candidate, try for a larger X
        else:
            right = mid - 1  # mid is too large, try a smaller X
    
    return left

class MultiResourceSemaphore:
    def __init__(self, total_resources: int):
        self.total_resources = total_resources
        self.available_resources = total_resources
        self.condition = threading.Condition()

    def acquire(self, requested: int):
        with self.condition:
            while self.available_resources < requested:
                self.condition.wait()
            self.available_resources -= requested

    def release(self, released: int):
        with self.condition:
            self.available_resources += released
            if self.available_resources > self.total_resources:
                self.available_resources = self.total_resources  # Prevent over-release
            self.condition.notify_all()

class ResourceManager():
    
    def __init__(self, n_threads: int, smart_policy: str = 'even', max_total_branches: int = 1000, max_n_samples: int = 100, policy_incremental_size=-1,policy_parallel_size=-1, mode: str = 'runout'):
        self.max_total_branches = max_total_branches
        self.max_tokens = [-1 for i in range(n_threads)]
        self.max_iterations = [-1 for i in range(n_threads)]
        if smart_policy == 'even':
            self.max_branches = [min(max_n_samples, self.max_total_branches // n_threads) for i in range(n_threads)]
        elif smart_policy == 'threshold' or smart_policy == 'threshold-one-shot' or smart_policy.startswith('smart'):
            self.max_branches = [min(max_n_samples, self.max_total_branches // n_threads) for i in range(n_threads)]
        else:   
            assert False
            self.max_branches = [-1 for i in range(n_threads)]

        self.idle_spin = [False for i in range(n_threads)]
        self.run_branches = [self.max_branches[i] for i in range(n_threads)]
        self.is_finished = [False for i in range(n_threads)]
        self.n_threads = n_threads

        self.program_entropy = [-1 for i in range(n_threads)]
        self.program_N_samples = [0 for i in range(n_threads)] 
        self.entropy_history = defaultdict(list)
        self.num_token_history = defaultdict(list)
        self.one_shot_flag = None 
        
        #self._lock = threading.Lock()
        self._is_dirty = [False for _ in range(n_threads)]

        self.policy = smart_policy
        self.mode = mode
        self.policy_incremental_size=policy_incremental_size
        self.policy_parallel_size=policy_parallel_size
        self.max_n_samples = max_n_samples

        self._lock = threading.Condition()

        # self.barrier = threading.Barrier(n_threads, timeout=5)

    def set_concurrent_programs(self, concurrent_programs):
        print('Setting concurrent programs', len(concurrent_programs))

        self.concurrent_programs = concurrent_programs
        self._init_program_resources()


    def update_program_entropy(self, program_id: int, entropy: float, N_samples: int):
        with self._lock:
            if N_samples != self.program_N_samples[program_id]:
                for i in self.concurrent_programs:
                    self._is_dirty[i] = True
                self._lock.notify_all()  
                
                self.program_entropy[program_id] = entropy
                self.entropy_history[program_id].append(entropy)
                self.program_N_samples[program_id] = N_samples 
        return

    #def update_program_finished(self, program_id: int, entropy: float, N_samples: int):
    #    with self._lock:
    #        self._is_dirty[program_id] = True
    #       self.is_finished[program_id] = True
    #        self.program_entropy[program_id] = entropy
    #        self.entropy_history[program_id].append(entropy)
    #        self.program_N_samples[program_id] = N_samples
    #    return

    def _init_program_resources(self):

        self.one_shot_flag = True
        #first allocate max
        self.concurrent_max_branches = sum(list(self.max_branches[i] for i in self.concurrent_programs))
        self.allocated_max_branches = 0
        
        if self.policy == 'even':
            for i in self.concurrent_programs:
                self.allocated_max_branches += self.max_branches[i]
        elif self.policy == 'smart' or self.policy.startswith('smart-strict'):
            for i in self.concurrent_programs:
                self.max_branches[i] = 5
                self.allocated_max_branches += 5
        elif self.policy == 'threshold':
            for i in self.concurrent_programs:
                self.max_branches[i] = 5
                self.allocated_max_branches += 5
        elif self.policy == 'threshold-one-shot':
            for i in self.concurrent_programs:
                self.max_branches[i] = 5
                self.allocated_max_branches += 5

        for i in self.concurrent_programs:
            self.run_branches[i] = max(0, min(self.max_n_samples, self.max_branches[i]) - self.program_N_samples[i])

        
        #if self.policy_incremental_size != -1:
        #    for i in self.concurrent_programs:
        #        self.run_branches[i] = min(self.run_branches[i], self.policy_incremental_size)
            #('Set: ', self.run_branches[i], self.policy_incremental_size)
        #print('init Total: ', self.max_branches, self.run_branches, self.policy_parallel_size,  self.policy_incremental_size)
        return

    def _redistribute_program_resources(self, program_id):
        #print('[XUT] Redistribute ', program_id)
        old_max_branches = self.max_branches[program_id]
        #Allocate Max Branches
        if self.policy == 'even':
            pass
            #print('Redistributing program resources evenly', self.max_total_branches, self.max_branches)
        elif self.policy.startswith('threshold'): # -one-shot':
            if self.mode == 'eco':
                if self.program_N_samples[program_id] >= 5 and self.max_branches[program_id] == 5:
                    H = self.program_entropy[program_id]
                    if H <= 0.5:                
                        self.max_branches[program_id] == 5
                    else:
                        new_max_branches = min(self.max_n_samples, self.max_total_branches // self.n_threads)         
                        #self.allocated_max_branches += new_max_branches - self.max_branches[program_id]
                        self.max_branches[program_id] = new_max_branches
            elif self.mode == 'runout':
                if self.program_N_samples[program_id] >= 5:
                    
                    if self.policy == 'threshold-one-shot':
                        H = self.entropy_history[program_id][0]
                    else:
                        H = self.program_entropy[program_id]

                    if H <= 0.5:           
                        self.max_branches[program_id] = max(self.program_N_samples[program_id], 5)
                        #if one shot, set 
                        if self.program_N_samples[program_id] == self.max_branches[program_id]:
                            self.is_finished[program_id] = True
                    else:
                        alloc_cnt = 0
                        not_alloc_cnt = 0
                        for i in self.concurrent_programs:
                            flag = i not in self.entropy_history or self.entropy_history[i][0] > 0.5  
                            alloc_cnt += flag
                            not_alloc_cnt += not flag
                            
                        new_max_branches = max(self.max_branches[program_id],min((self.concurrent_max_branches - 5 * not_alloc_cnt) // alloc_cnt, self.max_n_samples) )
                        new_max_branches = min(new_max_branches, self.max_branches[program_id] + self.concurrent_max_branches - self.allocated_max_branches )
                        
                        #print('X: ', new_max_branches, 5 * not_alloc_cnt, alloc_cnt, self.concurrent_max_branches)
                        
                        assert new_max_branches - self.max_branches[program_id] >= 0
                        #if new_max_branches - self.max_branches[program_id] > 0:
                        print('[OUT]', program_id, alloc_cnt, not_alloc_cnt, self.max_n_samples, self.max_branches[program_id], self.concurrent_max_branches, self.allocated_max_branches, new_max_branches, H)
                        
                        
                        self.max_branches[program_id] = new_max_branches

        elif self.policy.startswith('smart'):
            
            if self.mode == 'eco':                
                if self.program_N_samples[program_id] >= 5:
                    H = self.program_entropy[program_id]
                    D = 2.5
                    if H <= 0.5:
                        ret = 5
                    else:
                        ret = ( math.exp(H * math.log(2)) - 1) / (2 * 0.01 * D)   
                    print('Setting: ', ret, H, D, program_id)
                        
                    if self.program_N_samples[program_id] >= int(ret):                
                        
                        #new_max_branches = min(self.max_n_samples, max(self.program_N_samples[program_id], max(int(ret), self.max_branches[program_id])))
                        #new_max_branches = max(self.program_N_samples[program_id], new_max_branches)

                        self.max_branches[program_id] = max(int(ret), self.program_N_samples[program_id])
                        if self.program_N_samples[program_id] == self.max_branches[program_id]:
                            self.is_finished[program_id] = True

                    else:
                        new_max_branches = min(self.max_total_branches // self.n_threads, min(self.max_n_samples, max(int(ret), self.max_branches[program_id])))
                        new_max_branches = max(self.program_N_samples[program_id], new_max_branches)
                        
                        #new_max_branches = min(self.max_n_samples, self.max_total_branches // self.n_threads)         
                        #self.allocated_max_branches += new_max_branches - self.max_branches[program_id]
                        self.max_branches[program_id] = new_max_branches
                #print('SMART', program_id)
            elif self.mode == 'runout':
                if self.program_N_samples[program_id] >= 5:
                    
                    H = self.program_entropy[program_id]

                    if H <= 0.5:           
                        self.max_branches[program_id] = min(self.max_n_samples, max(self.program_N_samples[program_id], max(5, self.max_branches[program_id])))

                        if self.program_N_samples[program_id] == self.max_branches[program_id]:
                            self.is_finished[program_id] = True

                    else:
                        D = 1
                        '''
                        if self.policy.startswith('smart-strict'):
                            D = float(self.policy[len('smart-strict-'):])
                        else:
                            sum_ret = 0
                            ss = []
                            for i in self.concurrent_programs:
                                if i not in self.entropy_history:
                                    H = 10
                                else:
                                    H = self.program_entropy[i]
                                sum_ret += math.exp(H * math.log(2)) - 1
                                ss.append((H, math.exp(H * math.log(2)) - 1))
                            D = max(0.1, sum_ret / self.concurrent_max_branches / (2 * 0.01))
                        '''
                        
                        ret = ( math.exp(H * math.log(2)) - 1) / (2 * 0.01 * D)                    
                        new_max_branches = max(self.max_branches[program_id],min(int(ret), self.max_n_samples) )
                        new_max_branches = min(new_max_branches, self.max_branches[program_id] + self.concurrent_max_branches - self.allocated_max_branches )

                        max_delta = 5

                        new_max_branches = min(new_max_branches, self.max_branches[program_id] + max_delta)

                        assert new_max_branches - self.max_branches[program_id] >= 0
                        print('[OUT]', program_id, D, H, ret, self.max_n_samples, self.max_branches[program_id], self.concurrent_max_branches, self.allocated_max_branches, new_max_branches)
            
                        self.max_branches[program_id] = new_max_branches
                
        else:
            assert False, "Not Supported Scheduler"


        self.run_branches[program_id] = max(0, min(self.max_n_samples, self.max_branches[program_id]) - self.program_N_samples[program_id])

        if self.policy_incremental_size != -1:
            self.run_branches[program_id] = min(self.run_branches[program_id], self.policy_incremental_size)
        #if self.run_branches[program_id] == -1:
        assert self.run_branches[program_id] >= 0, f'Should not Happen {self.max_n_samples} {self.max_branches[program_id]} {self.program_N_samples[program_id]} {self.policy_incremental_size}'
        
        if self.policy == 'even' or (self.policy.startswith('threshold') and self.mode == 'eco') or (self.policy.startswith('smart') and self.mode == 'eco'):
            if self.program_N_samples[program_id] >= self.max_branches[program_id] \
                        and self.run_branches[program_id] == 0:
                self.is_finished[program_id] = True
        
        #print('POLICY: ', self.policy, self.max_branches, self.allocated_max_branches)
        
        self.allocated_max_branches += self.max_branches[program_id] - old_max_branches
        assert self.max_branches[program_id] - old_max_branches >= 0
        assert self.allocated_max_branches <= self.concurrent_max_branches
        if old_max_branches == self.max_branches[program_id] and self.run_branches[program_id] == 0:
            self.idle_spin[program_id] = True
        #elif self.is_finished[program_id]:
        #    self.idle_spin[program_id] = True
        else:
            self.idle_spin[program_id] = False

        
        return not self.idle_spin[program_id]
    
    def get_program_resources(self, program_id: int, run_branches: int) -> dict:
        with self._lock:
            should_terminate = self.should_program_terminated()
            #print('#Waiting: ', program_id, len(self._lock._waiters), should_terminate, sum(self.idle_spin), sum(self.is_finished), sum(self._is_dirty), run_branches)
            if run_branches == 0 and not self._is_dirty[program_id] and not should_terminate:
                #print('XWaiting: ', program_id, len(self._lock._waiters), should_terminate, self.idle_spin, self.is_finished, self._is_dirty, run_branches)

                print('[OUT] Waiting ', program_id, self._is_dirty[program_id], run_branches, self.max_branches[program_id])
                self._lock.wait()
            else:
                print('[OUT] Forwarding ', program_id, self._is_dirty[program_id], run_branches)
            
            #if self._is_dirty[program_id] or run_branches > 0:
            
                # Distribute the program status based on current entropy
            self._redistribute_program_resources(program_id)
            self._is_dirty[program_id] = False
                #pass
            print('[OUT] Redistributing program resources ', program_id, self.max_branches[program_id], self.run_branches[program_id])
        return dict(
            max_tokens=self.max_tokens[program_id],
            max_iterations=self.max_iterations[program_id],
            max_branches=self.max_branches[program_id],
            run_branches=self.run_branches[program_id],
        )

    def get_program_finished(self, program_id: int) -> bool:
        with self._lock:
            return self.is_finished[program_id]

    def should_program_terminated(self) -> bool:
        with self._lock:
            ret = all((self.idle_spin[i] or self.is_finished[i]) for i in self.concurrent_programs)
            #return self.concurrent_max_branches - self.allocated_max_branches <= 0
            if ret:
                self._lock.notify_all()
            return ret
                        

    def update_iteration_epilogue(self, program_id: int, entropy: float, N_samples: int, run_branches: int):
        #print('|wait| ', program_id, self.barrier.n_waiting, self.barrier.parties)
        #self.barrier.wait(timeout=None)
        #print('|dd| ', program_id, self.barrier.n_waiting, self.barrier.parties)        
        self.update_program_entropy(program_id, entropy, N_samples)
        
        #self.barrier.wait(timeout=None)
        #print('|xx| ', program_id, self.barrier.n_waiting, self.barrier.parties)
        resources = self.get_program_resources(program_id, run_branches) # immediately invoke redistribute resource after update program status

        #self.barrier.wait(timeout=None)
        #print('|zz| ', program_id, self.barrier.n_waiting, self.barrier.parties)
        return resources
    

import contextlib
@contextlib.contextmanager
def temp_fork(s: ProgramState):
    forks = s.fork(1)
    yield forks[0]
    forks.join()


from typing import Optional, Union

def gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    # return_logprob: Optional[bool] = None,
    return_logprob: Optional[bool] = True,
    logprob_start_len: Optional[int] = None,
    # top_logprobs_num: Optional[int] = None,
    top_logprobs_num: Optional[int] = 2,
    # return_text_in_logprobs: Optional[bool] = None,
    return_text_in_logprobs: Optional[bool] = True,
    dtype: Optional[Union[type, str]] = None,
    choices: Optional[List[str]] = None,
    choices_method: Optional['ChoicesSamplingMethod'] = None,
    regex: Optional[str] = None,
    json_schema: Optional[str] = None,
):
    return sgl.gen(
        name=name, 
        max_tokens=max_tokens, 
        min_tokens=min_tokens, 
        stop=stop, 
        stop_token_ids=stop_token_ids, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        min_p=min_p, 
        frequency_penalty=frequency_penalty, 
        presence_penalty=presence_penalty, 
        ignore_eos=ignore_eos, 
        return_logprob=return_logprob, 
        logprob_start_len=logprob_start_len, 
        top_logprobs_num=top_logprobs_num, 
        return_text_in_logprobs=return_text_in_logprobs, 
        dtype=dtype, 
        choices=choices, 
        choices_method=choices_method, 
        regex=regex, 
        json_schema=json_schema
    )
import os
fork_semaphore = MultiResourceSemaphore(int(os.environ.get("MAX_FORKS", 500)))

class NSampling:
    
    def __init__(
        self, 
        program_id: int = None, 
        resource_manager: ResourceManager = None,
        prompt: str = None,
        max_n_samples: int = 30,
        concurrent_programs: List[int] = None,
        policy_incremental_size: int = 1,
        policy_parallel_size: int = -1,
    ):
        self.program_id = program_id
        self.resource_manager = resource_manager
        self.prompt = prompt


        self.tokens = []
        self.answers = []
        self.extracted_answers = []
        self.concurrent_programs = concurrent_programs
        self.policy_incremental_size = policy_incremental_size
        self.policy_parallel_size = policy_parallel_size
        self.start_time = time.time()
        self.end_time = None
        pass

    @staticmethod
    def prepare_batch(
        resource_manager, 
        max_n_samples:int=30, 
        arguments:'List'=None,
        chunk_size: int = -1,
        policy_incremental_size: int = -1,
        policy_parallel_size: int = -1,
    ):
        n_programs = len(arguments)
        programs = []
        for i in range(n_programs):
            
            assert chunk_size > 0
            concurrent_programs = [i // chunk_size * chunk_size + j for j in range(chunk_size)]
            
            program = NSampling(
                program_id=i, 
                resource_manager=resource_manager, 
                max_n_samples=max_n_samples,
                concurrent_programs=concurrent_programs,
                policy_incremental_size=policy_incremental_size,
                policy_parallel_size=policy_parallel_size,
                **arguments[i]
            )
            programs.append(dict(program=program))

        return programs


    def run(self, s: ProgramState, resource_manager: ResourceManager, iter_num: int, run_branches: int):
        if iter_num == 0:
            s += system("You are a helpful assistant.")
            s += user(self.prompt + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.")
            self.real_prompt = s.text()

        # self.accumulated_n_samples
        # TODO - Hopefully we can just bind the program_id to the resource manager as a functor...
        #resources = resource_manager.get_program_resources(self.program_id)
        # max_tokens = resources['max_tokens']
        # max_iterations = resources['max_iterations']

        #run_branches = resources['run_branches']
    
        print(f"[{self.program_id}] Running iteration {iter_num} with {run_branches} branches.")
        if run_branches > 0:
            forks = s.fork(run_branches)
            for f in forks:
                f += assistant(gen("answer", max_tokens=512, return_logprob=True, temperature=0.7))
            forks.join()

                
            print(f"[{self.program_id}] Joined forks.")
            
            for f in forks:
                answer = f.get_var("answer")
                self.answers.append(answer)
                self.extracted_answers.append(extract_answer(answer))
                self.tokens.append(f.get_meta_info("answer")['completion_tokens'])
        print(f"[{self.program_id}] Collected answers.")
        
        entropy = item_entropy(self.extracted_answers)
        N_samples = len(self.extracted_answers)

        #self.accumulated_n_samples += run_branches
        #assert self.accumulated_n_samples == N_samples

        #Note questions will be killed [FOREVER] here
        #if self.accumulated_n_samples >= self.max_n_samples:
        #    print(f"[{self.program_id}] Reached max_n_samples, terminating.")
        #    resource_manager.update_program_finished(self.program_id, entropy, N_samples)

        return entropy, N_samples

Finished = [0]
@sgl.function
def runner(s: ProgramState, program: NSampling=None):
    resource_manager: ResourceManager = program.resource_manager
    iter_num = 0
    

    resources = resource_manager.get_program_resources(program.program_id, -1) #(program.program_id, entropy, N_samples, -1)
    
    while not resource_manager.get_program_finished(program.program_id):
        print(f"[{program.program_id}] Iteration {iter_num}")
        #is_finished = resource_manager.get_program_finished(program.program_id)
        #if is_finished:   
        #break
        is_terminated = resource_manager.should_program_terminated()
        if is_terminated:
            break
        #program.end_time = time.time()
        #    break
        #if not is_finished:
        #resources = resource_manager.get_program_resources(program.program_id)
        # max_tokens = resources['max_tokens']
        # max_iterations = resources['max_iterations']

        run_branches = resources['run_branches']

        #if run_branches == 0:
        #for _ in range(run_branches):
        fork_semaphore.acquire(run_branches)

        print(f"[{program.program_id}] Running iteration {iter_num}")
        entropy, N_samples = program.run(s, resource_manager, iter_num, run_branches)
        print(f"[{program.program_id}] Iteration {iter_num} finished.")

        #for _ in range(run_branches):
        fork_semaphore.release(run_branches)

        #update allocate here if status is changed
        resources = resource_manager.update_iteration_epilogue(program.program_id, entropy, N_samples, run_branches)
 
        iter_num += 1
        if iter_num >= 1000:
            iter_num = iter_num % 1000
            
    program.end_time = time.time()
    print('Finished ', sum(resource_manager.is_finished), Finished[0] + 1)
    Finished[0] += 1
    return program.program_id

def extract_groundtruth(groundtruth_str: str):
    x = groundtruth_str.split("#### ")[1].strip().replace(",", "")
    try:
        float(x)
    except:
        raise ValueError(
            "Warning: Error should raise since the extracted groundtruth string {}\
             cannot be converted to float".format(
                x
            )
        )
    return x

from utils import load_jsonl



def n_sampling_interface(
    n_test_cases: int,
    chunk_size: int,
    max_n_samples: int,
    voting: str = 'maj',
    outfile: str = 'gsm.json',
    stat_output_path: str = 'gsm.log',
    mode: str = 'runout',

    policy_incremental_size: int = -1,
    policy_parallel_size: int = -1,
    smart_policy: str = 'even',
    max_total_branches: int = 1000,
):
    voting_method = majority_voting if voting == 'maj' else majority_voting

    data = load_jsonl('data-sc/GSM8K/test.jsonl')
    random.seed(42)
    random.shuffle(data)

    arguments = [
        dict(prompt=data[i]['question']) for i in range(n_test_cases)
    ]
    resource_manager = ResourceManager(n_threads=len(arguments), smart_policy=smart_policy, max_total_branches=max_total_branches, policy_incremental_size=policy_incremental_size,
                                      policy_parallel_size=policy_parallel_size, max_n_samples=max_n_samples, mode=mode)

    if chunk_size == -1:
        chunk_size = len(arguments)
    programs = NSampling.prepare_batch(resource_manager, max_n_samples=max_n_samples, arguments=arguments, chunk_size=chunk_size, policy_incremental_size=policy_incremental_size,
                                      policy_parallel_size=policy_parallel_size)

    start_time = time.time()
    def chunk_iter(iterable, size=chunk_size):
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, size))
            if not chunk:
                break
            yield chunk

    for chunk in chunk_iter(programs, size=chunk_size):
        resource_manager.set_concurrent_programs(concurrent_programs=chunk[0]['program'].concurrent_programs)
        states = runner.run_batch(chunk, num_threads=len(chunk), progress_bar=True)
        #print(len(states), states[0].ret_value)

    end_time = time.time()
    ground_truths = []
    latency = []
    save_infos = []
    success_items = 0
    for i, program in enumerate(programs):
        gt = extract_groundtruth(data[i]['answer'])
        ground_truths.append(gt)
        extracted_answers = program['program'].extracted_answers
        res = math_equal(gt, voting_method(extracted_answers))
        success_items += res
        latency.append(program['program'].end_time - program['program'].start_time)
        save_infos.append({'id' : program['program'].program_id, 'answers' : program['program'].answers, 'tokens' : program['program'].tokens, 'extracted_answers' : program['program'].extracted_answers, 'gt' : gt, 'latency' : latency[-1], 'prompt' : program['program'].real_prompt})

    with open(outfile, 'w') as f:
        for info in save_infos:
            json.dump(info, f, indent=4)
            f.write('\n')

    print()
    with open(stat_output_path, "a") as stat_file:
        for f in [stat_file, sys.stdout]:    
            print(f"Self-consistency GSM8K Summary", file=f)
            print(f"Policy: {smart_policy}", file=f)
            print(f"Policy incremental size: {policy_incremental_size}", file=f)
            print(f"Policy Parallel size: {policy_parallel_size}", file=f)
            print(f"Max total branches: {max_total_branches}", file=f)
            print(f"Real Max total branches: {sum(resource_manager.max_branches)}", file=f)
            print(f"Real Total branches: {sum(resource_manager.program_N_samples)}", file=f)

            print(f"Real Max total branches in list: {resource_manager.max_branches}", file=f)
            print(f"Real Total branches in list: {resource_manager.program_N_samples}", file=f)

            print(f"Max n samples: {max_n_samples}", file=f)
            print(f"Chunk size: {chunk_size}", file=f)
            print(f"N test cases: {n_test_cases}", file=f)
            print(f"============================", file=f)
            print(f"Accuracy: {success_items / n_test_cases}", file=f)
            print(f"Average latency: {sum(latency) / n_test_cases}", file=f)
            print(f"Generated tokens: {sum(sum(tokens) for tokens in [p['program'].tokens for p in programs])}", file=f)
            #print(f"Sample answers: {programs[0]['program'].extracted_answers}", file=f)
            print(f"Time cost: {end_time - start_time}", file=f)
            print(file=f)


def test_n_sampling(
    n_test_cases: int,
    chunk_size: int,
    max_n_samples: int,
    voting: str = 'maj',
    outfile: str = 'gsm.json',
    stat_output_path: str = 'gsm.log',
    mode: str = 'runout',

    policy_incremental_size: int = -1,
    policy_parallel_size: int = -1,
    smart_policy: str = 'even',
    max_total_branches: int = 1000,
):
    voting_method = majority_voting if voting == 'maj' else majority_voting

    data = load_jsonl('data-sc/GSM8K/test.jsonl')
    random.seed(42)
    random.shuffle(data)

    arguments = [
        dict(prompt=data[i]['question']) for i in range(n_test_cases)
    ]
    resource_manager = ResourceManager(n_threads=len(arguments), smart_policy=smart_policy, max_total_branches=max_total_branches, policy_incremental_size=policy_incremental_size,
                                      policy_parallel_size=policy_parallel_size, max_n_samples=max_n_samples, mode=mode)

    if chunk_size == -1:
        chunk_size = len(arguments)
    programs = NSampling.prepare_batch(resource_manager, max_n_samples=max_n_samples, arguments=arguments, chunk_size=chunk_size, policy_incremental_size=policy_incremental_size,
                                      policy_parallel_size=policy_parallel_size)

    start_time = time.time()
    def chunk_iter(iterable, size=chunk_size):
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, size))
            if not chunk:
                break
            yield chunk

    for chunk in chunk_iter(programs, size=chunk_size):
        resource_manager.set_concurrent_programs(concurrent_programs=chunk[0]['program'].concurrent_programs)
        states = runner.run_batch(chunk, num_threads=len(chunk), progress_bar=True)
        #print(len(states), states[0].ret_value)

    end_time = time.time()
    ground_truths = []
    latency = []
    save_infos = []
    success_items = 0
    for i, program in enumerate(programs):
        gt = extract_groundtruth(data[i]['answer'])
        ground_truths.append(gt)
        extracted_answers = program['program'].extracted_answers
        res = math_equal(gt, voting_method(extracted_answers))
        success_items += res
        latency.append(program['program'].end_time - program['program'].start_time)
        save_infos.append({'id' : program['program'].program_id, 'answers' : program['program'].answers, 'tokens' : program['program'].tokens, 'extracted_answers' : program['program'].extracted_answers, 'gt' : gt, 'latency' : latency[-1], 'prompt' : program['program'].real_prompt})

    with open(outfile, 'w') as f:
        for info in save_infos:
            json.dump(info, f, indent=4)
            f.write('\n')

    print()
    with open(stat_output_path, "a") as stat_file:
        for f in [stat_file, sys.stdout]:    
            print(f"Self-consistency GSM8K Summary", file=f)
            print(f"Policy: {smart_policy}", file=f)
            print(f"Policy incremental size: {policy_incremental_size}", file=f)
            print(f"Policy Parallel size: {policy_parallel_size}", file=f)
            print(f"Max total branches: {max_total_branches}", file=f)
            print(f"Real Max total branches: {sum(resource_manager.max_branches)}", file=f)
            print(f"Real Total branches: {sum(resource_manager.program_N_samples)}", file=f)

            print(f"Real Max total branches in list: {resource_manager.max_branches}", file=f)
            print(f"Real Total branches in list: {resource_manager.program_N_samples}", file=f)

            print(f"Max n samples: {max_n_samples}", file=f)
            print(f"Chunk size: {chunk_size}", file=f)
            print(f"N test cases: {n_test_cases}", file=f)
            print(f"============================", file=f)
            print(f"Accuracy: {success_items / n_test_cases}", file=f)
            print(f"Average latency: {sum(latency) / n_test_cases}", file=f)
            print(f"Generated tokens: {sum(sum(tokens) for tokens in [p['program'].tokens for p in programs])}", file=f)
            #print(f"Sample answers: {programs[0]['program'].extracted_answers}", file=f)
            print(f"Time cost: {end_time - start_time}", file=f)
            print(file=f)

    #print(programs[0]['program'].extracted_answers)
    #print(sum(programs[0]['program'].tokens))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process cmds.')
    parser.add_argument('--n_test_cases', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--chunk_size', type=int, default=-1, help='Number of samples to generate')
    parser.add_argument('--max_n_samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--voting', type=str, default='maj', help='Number of samples to generate')
    parser.add_argument('--outfile', type=str, default='gsm.json', help='Number of samples to generate')
    parser.add_argument('--stat_output_path', type=str, default='gsm.log', help='Number of samples to generate')
    parser.add_argument('--max_total_branches', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--port', type=int, default=30000, help='Number of samples to generate')
    parser.add_argument('--mode', type=str, default='runout', choices=['runout', 'eco'], help='Number of samples to generate')

    parser.add_argument('--policy_incremental_size', type=int, default=-1, help='Number of samples to generate')
    parser.add_argument('--policy_parallel_size', type=int, default=-1, help='Number of samples to generate')
    parser.add_argument('--smart_policy', type=str, default='even', help='Number of samples to generate')

    args = parser.parse_args()

    llm = RuntimeEndpoint(f"http://localhost:{args.port}")
    set_default_backend(llm)

    test_n_sampling(
        n_test_cases = args.n_test_cases,
        chunk_size = args.chunk_size,
        max_n_samples = args.max_n_samples,
        voting = args.voting,
        outfile = args.outfile,
        stat_output_path = args.stat_output_path,
        max_total_branches = args.max_total_branches,
        mode = args.mode,

        policy_parallel_size = args.policy_parallel_size,
        policy_incremental_size = args.policy_incremental_size,
        smart_policy = args.smart_policy,
    )