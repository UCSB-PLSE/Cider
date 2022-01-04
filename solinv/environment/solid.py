import subprocess
import json
import re

from lark.exceptions import *

from .contract import Contract
from ..tyrell.spec import sort

class Solid:
  def __init__(self):
    pass

  def check(self, contract: Contract, invariant_str):
    """Check whether the invariant folds for a contract"""
    cmd = f"liquidsol-exe {contract.path} --task check --check-inv '{invariant_str}' --only-last"
    ret = subprocess.run( cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
      raise ValueError(ret.stderr.decode("utf-8"))
    ret = ret.stdout.decode("utf-8")
    hard_ok, hard = map(int, re.search("Hard: ([0-9]+) / ([0-9]+)", ret).groups())
    soft_ok, soft = map(int, re.search("Soft: ([0-9]+) / ([0-9]+)", ret).groups())
    return hard_ok, hard, soft_ok, soft
  
  def use_solc_version(self, version_str):
    """Use the specified version of solc"""
    cmd = f"solc-select use {version_str}"
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        raise Exception("Error executing solc-select. Check your environment configuration.")
  
  def contract_json(self, contract: Contract):
    """Get the contract AST as a json dict"""
    cmd = f"liquidsol-exe {contract.path} --task ast --only-last"
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
      raise Exception(f"Error executing {cmd}. Check your environment configuration.")
    remove_first_line = lambda s: '\n'.join(s.split('\n')[1:])
    raw_output = ret.stdout.decode("utf-8")
    raw_json = remove_first_line(raw_output)
    parsed_json = json.loads(raw_json)
    return parsed_json
  
  def storage_variables(self, contract: Contract):
    """Get the list of storage variables as well as a list of their sorts"""
    cmd = f"liquidsol-exe {contract.path} --only-last --task vars"
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
      raise Exception(f"Error executing {cmd}. Check your environment configuration.\n{ret.stderr.decode('utf-8')}")
    raw_output = ret.stdout.decode("utf-8")
    lines = raw_output.rstrip().split("\n")[1:] # first line is "Now running on ..."
    variables, sorts = [], []
    # split different classes
    for line in lines:
      name_str, sort_str = line.strip().split(" : ")
      try:
        s = sort.parse(sort_str)
        variables.append(name_str)
        sorts.append(s)
      except: # fixme: more precise exception matching
        # raise Exception("Unable to parse sort {}".format(sort_str))
        print(f"Warning: Unable to parse sort {sort_str}. Discard variable {name_str}")
    # fixme: remove duplicate, this is not super appropriate
    # tmp_list = list(set(tmp_list))
    # print("# number of stovars: {}, stovars are: {}, sorts: {}".format(len(variables), variables, sorts))
    return variables, sorts
  
  def flow(self, contract: Contract):
    """Get the JSON of storage-variable graph"""
    cmd = f"liquidsol-exe {contract.path} --only-last --task flow"
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
      raise Exception(f"Error executing {cmd}. Check your environment configuration.\n{ret.stderr.decode('utf-8')}")
    return json.loads(ret.stdout.decode("utf-8").split("\n")[1]) # first line is "Now running on ..."