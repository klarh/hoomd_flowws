
## Introduction

`hoomd_flowws` is an in-development set of modules to create reusable
scientific workflows using
[hoomd-blue](https://github.com/glotzerlab/hoomd-blue). While the
python API of hoomd-blue holds enormous possibility for scriptability
(including making projects like this possible in the first place),
this flexibility can also lead to poorly-structured, rigid script
workflows if not carefully managed. The aim of this project is to
formulate a set of robust, modular individual components that can be
composed to perform most common workflows.

`hoomd-flowws` is being developed in conjunction with
[flowws](https://github.com/klarh/flowws).

## Installation

Install hoomd_flowws from source using `pip`:

```
# install flowws first
git clone https://github.com/klarh/flowws
pip install flowws

git clone https://github.com/klarh/hoomd_flowws
pip install hoomd_flowws
```
