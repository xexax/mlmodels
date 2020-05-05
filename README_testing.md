# Testing Mechanism
### Pull Requests, Fixes, New Models
Read following instructions before adding a new model.

- [MANDATORY For TESTS](#configure-for-tests)



### Overview

  Configuration is here :
     https://github.com/arita37/mlmodels/blob/dev/.github/workflows/



####  Code for testing is here :
      https://github.com/arita37/mlmodels/blob/dev/mlmodels/ztest.py



####  Helper code
```python
     from mlmodels.util import os_package_root_path, log, 
                            path_norm, get_model_uri, path_norm_dict

     ### Use path_norm to normalize your path.
     data_path = path_norm("dataset/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/dataset/text/myfile.txt


     ### Use path_norm to normalize your path.
     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt


     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt
```


