class Contract:
  def __init__(self, contract_path, solc_version):
    self._path = contract_path
    self._version = solc_version
  
  @property
  def path(self):
    return self._path
  
  @property
  def version(self):
    return self._version