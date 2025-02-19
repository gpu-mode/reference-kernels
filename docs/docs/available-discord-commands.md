---
sidebar_position: 5
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Available Discord Commands
We advise that you read this section carefully -- it is short, but will be extremely useful for you
to remember. These `/` commands allow you to interface with the leaderboard, as well as extract any
relevant information you may need. We allow participants to view all aspects of the evaluation
pipeline, including leaderboard reference code and our evaluation scripts. We separate the commands
into three categories, which you can open in the tabs below.


<Tabs>
  <TabItem value="submission" label="Leaderboard Submission Commands" default>
        ## Submission Commands

        The following commands allow participants to submit their kernel code to a specific GPU on a
        specific leaderboard. We currently support two runners, **GitHub** and **Modal**, which each contain their
        own set of available GPUs. After running a submission command, **a UI selection window will pop up asking the
        participant to select which GPUs to submit to** (multiple can be selected at once).

        ---
        ### `/leaderboard submit ranked {script}`
        **Description:** Submit a kernel to an official leaderboard. Depending on the leaderboard, you can
        submit to a subset of `AMD MI300`, `NVIDIA T4`,
        `NVIDIA L4`, `NVIDIA A100`, and `NVIDIA H100` GPUs.

        **Arguments:**
        - `script` *(required)*: Script to be submitted. Note, a Python leaderboard expects a Python
        submission file, and a CUDA leaderboard expects a CUDA submission file.
        - `leaderboard_name` *(optional)*: Name of the leaderboard to submit to - If specified in the submission
        heading with `!POPCORN leaderboard {name}`, this is not required.
        - `gpu` *(optional)*: Specify a GPU to submit to. If not specified, the bot will prompt the participant
        to select GPU(s) to submit to.

        ---
        ### `/leaderboard submit test {script}`
        **Description:** Check the functional correctness of a kernel on a specific leaderboard. 
        Does not make an official submission!
        Depending on the leaderboard, you can
        submit to a subset of `AMD MI300`, `NVIDIA T4`,
        `NVIDIA L4`, `NVIDIA A100`, and `NVIDIA H100` GPUs.

        **Arguments:**
        - `script` *(required)*: Script to be submitted. Note, a Python leaderboard expects a Python
        submission file, and a CUDA leaderboard expects a CUDA submission file.
        - `leaderboard_name` *(optional)*: Name of the leaderboard to submit to - If specified in the submission
        heading with `!POPCORN leaderboard {name}`, this is not required.
        - `gpu` *(optional)*: Specify a GPU to submit to. If not specified, the bot will prompt the participant
        to select GPU(s) to submit to.

        ---
        ### `/leaderboard submit benchmark {script}`
        **Description:** Benchmark the speed of a kernel on a specific leaderboard. 
        Does not make an official submission!
        Depending on the leaderboard, you can
        submit to a subset of `AMD MI300`, `NVIDIA T4`,
        `NVIDIA L4`, `NVIDIA A100`, and `NVIDIA H100` GPUs.

        **Arguments:**
        - `script` *(required)*: Script to be submitted. Note, a Python leaderboard expects a Python
        submission file, and a CUDA leaderboard expects a CUDA submission file.
        - `leaderboard_name` *(optional)*: Name of the leaderboard to submit to - If specified in the submission
        heading with `!POPCORN leaderboard {name}`, this is not required.
        - `gpu` *(optional)*: Specify a GPU to submit to. If not specified, the bot will prompt the participant
        to select GPU(s) to submit to.


  </TabItem>
  <TabItem value="tools" label="Useful Info Commands">

        ## Useful Info Commands
        These commands are particularly useful for leaderboard participants to query for information
        about the leaderboards that they are submitting to. For example, listing the available
        leaderboards to submit to, viewing the source code for the reference kernel, etc.

        ---

        ### `/leaderboard list`
        **Description:** Lists all available leaderboards. Each leaderboard shows its name,
        deadline, and all GPUs that can be submitted to on that leaderboard.

        **Arguments:** None.

        *Example output*:
        ```rust title="/leaderboard list"
        Name                    Deadline             GPU Types     
        --------------------------------------------------------------------------------
        softmax_py              2025-01-31 00:00     NVIDIA T4, AMD MI250
        softmax_cuda            2024-12-31 00:00     NVIDIA T4
        matmul_cuda             2025-12-12 00:00     NVIDIA T4, NVIDIA A100, NVIDIA H100
        llama3-inference_py     2025-12-31 00:00     NVIDIA A100, NVIDIA H100
        ```
        For example, for the `llama3-inferece_py` leaderboard, users may only submit to `NVIDIA
        A100` and `NVIDIA H100` GPUs.

        ---

        ### `/leaderboard show {leaderboard_name}`
        **Description:** Display rankings for a particular leaderboard. After running this command,
        a UI will pop up asking the user which GPUs for that particular leaderboard should be displayed (can select multiple).


        **Arguments:** 
        - `leaderboard_name` *(required)*: Name of the leaderboard to show information for.

        ---

        ### `/leaderboard show-personal {leaderboard_name}`
        **Description:** Display rankings of **your own submissions** for a particular leaderboard. After running this command,
        a UI will pop up asking the user which GPUs for that particular leaderboard should be displayed (can select multiple).

        **Arguments:** 
        - `leaderboard_name` *(required)*: Name of the leaderboard to show information for.

        ---

        ### `/leaderboard task {leaderboard_name}`
        **Description:** Each leaderboard has a set of functions, including the reference kernel,
        the evaluation harness, and task specifications, that the leaderboard creator implemented. 
        As a participant, you can view all of these code files using this command.

        **Arguments:** 
        - `leaderboard_name` *(required)*: Name of the leaderboard to retrieve reference code for.



  </TabItem>
  <TabItem value="creation" label="Leaderboard Creation Commands">

        ## Leaderboard Creation Commands
        The following commands allow leaderboard creators to specify algorithms / kernels that they
        want participants to compete on. Only Discord users with the `Leaderboard Admin` or `Leaderboard
        Creator` role can run these commands on a particular server such as GPU MODE. All reference
        code must satisfy a particular format outlined in [Creating a Leaderboard](#).

        After running a creation command, **a UI selection window will pop up asking the
        creator to select which GPUs this leaderboard should run on** (multiple can be selected at once).

        ---

        ### `/leaderboard create`
        **Description:** Creates a new leaderboard according to the defined reference code. Each
        leaderboard uniquely defines the input data to evaluate on, a reference kernel to compare user
        submissions to, and a comparison function that checks that two outputs match. Most standard
        kernels will assume a list of Tensors, but we enable this flexibility for leaderboard creators to
        define anything as the input and output types.

        **Arguments:** 
        - `leaderboard_name` *(required)*: Name of the leaderboard to create.
        - `deadline` *(required)*: When the leaderboard finishes. Must be of the form YYYY-MM-DD.
        - `task_zip` *(required)*: The evaluation harness (as a .zip) that defines the leaderboard. This folder
        must specify a `task.yml` and the relevant files, which we detail in [Creating a Leaderboard](./creating-a-leaderboard/cuda-creations),
        and its file extension (e.g. `.py` or `.cu`) determines whether the leaderboard submissions must be
        in Python or CUDA/C++.

        ---

        ### `/leaderboard delete {leaderboard_name}`
        **Description:** ⚠️  Delete an existing leaderboard and all submissions information to that
        leaderboard. Use with extreme care -- a message will pop up asking the user to verify that they want
        to do this.

        **Arguments:** 
        - `leaderboard_name` *(required)*: Name of the leaderboard to delete.
  </TabItem>
</Tabs>


