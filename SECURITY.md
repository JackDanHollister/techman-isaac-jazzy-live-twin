# Security and safety reporting

Please report software-security issues privately through GitHub's security
advisory feature rather than opening a public issue.

Robot-motion or stop-verification defects should be treated as safety-relevant:

- do not reproduce them with a physical robot unless the cell and vendor safety
  system are independently controlled;
- preserve the mode, report, software revision, controller/TMflow version, and
  exact failure text;
- do not include credentials, tokens, private keys, or credential-bearing URLs.

Isaac Sim is not a safety controller. The physical E-stop and the robot's
certified safety system remain authoritative.
