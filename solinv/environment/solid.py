import subprocess
import json
import re
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
    cmd = f"liquidsol-exe {contract.path} --task vars --only-last"
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
      raise Exception(f"Error executing {cmd}. Check your environment configuration.")
    raw_output = ret.stdout.decode("utf-8")
    lines = raw_output.rstrip().split("\n")
    # split different classes
    break_points = [i for i, line in enumerate(lines) if line.startswith("Now running")]
    break_points.append(len(lines))
    # process every block
    variables = []
    sorts = []
    for (bp0, bp1) in zip(break_points, break_points[1:]):
      curr_lines = lines[bp0+1 : bp1]
      print(curr_lines)
      assert(len(curr_lines) % 2 == 0)
      names = curr_lines[:len(curr_lines)//2]
      sorts_str = curr_lines[len(curr_lines)//2:]
      for k in range(len(names)):
        try:
          s = sort.parse(sorts_str[k])
          variables.append(names[k])
          sorts.append(s)
        except:
          # print("Warning: unable to parse sort {}".format(sorts[k]))
          raise Exception("unable to parse sort {}".format(sorts_str[k]))
    # fixme: remove duplicate, this is not super appropriate
    # tmp_list = list(set(tmp_list))
    print("# number of stovars: {}, stovars are: {}, sorts: {}".format(len(variables), variables, sorts))
    return variables, sorts