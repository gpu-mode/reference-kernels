---
sidebar_position: 4
---

# (Optional) Submitting using the CLI
We also provide an optional command line interface (CLI), which can be **used instead of the Discord bot to
submit to leaderboards**. In this page, we will walk through setting up the [Popcorn CLI](https://github.com/gpu-mode/popcorn-cli).


## Setup
The [Popcorn CLI GitHub](https://github.com/gpu-mode/popcorn-cli) provides instructions for setting up the CLI. In this walkthrough,
we will use the release binaries that are downloadable [here](https://github.com/gpu-mode/popcorn-cli/releases). First, download
the `.zip` relevant to your computer and extract it into a suitable location.

We can either directly run the binary and add it to our path using (fill in `popcorn_cli_path` with your own)
<center>
```
export PATH="{popcorn_cli_path}:$PATH"
```
</center>

To make this permanent, you can also add it to your `~/.bashrc` / `~/.bash_profile` / `~/.zshrc`.

## An Example Walkthrough
We have to point our CLI to the Popcorn API in our Discord bot. 

<center>
```
export POPCORN_API_URL={API_URL}
```
</center>

The rest of the CLI is self-explanatory -- you can submit files to specific leaderboards like in the previous sections
using 
<center>
```
popcorn-cli <submission-file>
```
</center>

