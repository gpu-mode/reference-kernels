# discord-cluster-manager

This is the code for the Discord bot we'll be using to queue jobs to a cluster of GPUs that our generous sponsors have provided. Our goal is to be able to queue kernels that can run end to end in seconds that way things feel interactive and social.

The key idea is that we're using Github Actions as a job scheduling engine and primarily making the Discord bot interact with the cluster via issuing Github Actions and and monitoring their status and while we're focused on having a nice user experience on discord.gg/gpumode, [we're happy to accept PRs](#local-development) that make it easier for other Discord communities to hook GPUs.

[Demo!](https://www.youtube.com/watch?v=-u7kX_vpLfk)

## Table of Contents

- [Supported Schedulers](#supported-schedulers)
- [Local Development](#local-development)
  - [Clone Repository](#clone-repository)
  - [Discord Bot](#discord-bot)
  - [Database](#database)
  - [Environment Variables](#environment-variables)
  - [Verify Setup](#verify-setup)
- [Available Commands](#available-commands)
- [Using the Leaderboard](#using-the-leaderboard)
  - [Creating a New Leaderboard](#creating-a-new-leaderboard)
    - [Reference Code Requirements (Python)](#reference-code-requirements-python)
    - [Reference Code Requirements (CUDA)](#reference-code-requirements-cuda)
  - [Submitting to a Leaderboard](#submitting-to-a-leaderboard)
  - [Other Available Leaderboard Commands](#other-available-leaderboard-commands)
    - [GPU Kernel-Specific Commands](#gpu-kernel-specific-commands)
- [Testing the Discord Bot](#testing-the-discord-bot)
- [How to Add a New GPU to the Cluster](#how-to-add-a-new-gpu-to-the-cluster)
- [Acknowledgements](#acknowledgements)

## Supported schedulers

- GitHub Actions
- Modal
- Slurm (not implemented yet)

## Local Development

### Clone Repository

> [!IMPORTANT]
> Do not fork this repository. Instead, directly clone this repository to your local machine.

> [!IMPORTANT]
> Python 3.11 or higher is required.

After, install the dependencies with `pip install -r requirements.txt`.

### Setup Discord Bot

To run and develop the bot locally, you need to add it to your own "staging" server. Follow the steps [here](https://discordjs.guide/preparations/setting-up-a-bot-application.html#creating-your-bot) and [here](https://discordjs.guide/preparations/adding-your-bot-to-servers.html#bot-invite-links) to create a bot application and then add it to your staging server.

Below is a visual walk-through of the steps linked above:

- The bot needs the `Message Content Intent` and `Server Members Intent` permissions turned on.
  <details>
    <summary>Click here for visual.</summary>
    <img width="1440" alt="DCS_bot_perms" src="https://github.com/user-attachments/assets/31ee441a-f8a9-4a2f-89d1-fda171947bfd" />
  </details>

- The bot needs `applications.commands` and `bot` scopes.

  <details>
      <summary>Click here for visual.</summary>
    <img width="1440" alt="Screenshot 2024-11-24 at 12 34 09 PM" src="https://github.com/user-attachments/assets/31302214-1d5a-416a-b7b4-93a44442be51">
  </details>

- Finally, generate an invite link for the bot and enter it into any browser.

  <details>
      <summary>Click here for visual.</summary>
      <img width="1440" alt="Screenshot 2024-11-24 at 12 44 08 PM" src="https://github.com/user-attachments/assets/54c34b6b-c944-4ce7-96dd-e40cfe79ffb3">
  </details>

> [!NOTE]
> Bot permissions involving threads/mentions/messages should suffice, but you can naively give it `Administrator` since it's just a test bot in your own testing Discord server.

### Database

The leaderboard persists information in a Postgres database. To develop locally, set Postgres up on your machine. Then start a Postgres shell with `psql`, and create a database:

```
$ psql -U postgres
Password for user postgres: ********
psql (16.6 (Ubuntu 16.6-1.pgdg22.04+1))
Type "help" for help.

postgres=# CREATE DATABASE clusterdev;
```

We are using [Yoyo Migrations](https://ollycope.com/software/yoyo/) to manage tables, indexes, etc. in our database. To create tables in your local database, apply the migrations in `src/discord-cluster-manager/migrations` with the following command line:

```
yoyo apply src/discord-cluster-manager/migrations \
  -d postgresql://user:password@localhost/clusterdev
```

  <details>
    <summary>Click here for a transcript of a yoyo apply session</summary>

    $ yoyo apply . -d postgresql://user:password@localhost/clusterdev

    [20241208_01_p3yuR-initial-leaderboard-schema]
    Shall I apply this migration? [Ynvdaqjk?]: y

    Selected 1 migration:
      [20241208_01_p3yuR-initial-leaderboard-schema]
    Apply this migration to postgresql://user:password@localhost/clusterdev [Yn]: y
    Save migration configuration to yoyo.ini?
    This is saved in plain text and contains your database password.

    Answering 'y' means you do not have to specify the migration source or database connection for future runs [yn]: n

  </details>

Applying migrations to our staging and prod environments also happens using `yoyo apply`, just with a different database URL.

To make changes to the structure of the database, create a new migration:

```
yoyo new src/discord-cluster-manager/migrations -m "short_description"
```

...and then edit the generated file. Please do not edit existing migration files: the existing migration files form a sort of changelog that is supposed to be immutable, and so yoyo will refuse to reapply the changes.

We are following an expand/migrate/contract pattern to allow database migrations without downtime. When you want to make a change to the structure of the database, first determine if it is expansive or contractive.

- _Expansive changes_ are those that have no possibility of breaking a running application. Examples include: adding a new nullable column, adding a non-null column with a default value, adding an index, adding a table, etc.
- _Contractive changes_ are those that could break a running application. Examples include: dropping a table, dropping a column, adding a not null constraint to a column, adding a unique index, etc.

After an expansive phase, data gets migrated to the newly added elements. Code also begins using the newly added elements. This is the migration step. Finally, when all code is no longer using elements that are obsolete, these can be removed. (Or, if adding a unique or not null constraint, after checking that the data satisfies the constraint, then the constraint can be safely added.)

Expand, migrate, and contract steps may all be written using yoyo.

### Environment Variables

Create a `.env` file with the following environment variables:

- `DISCORD_DEBUG_TOKEN` : The token of the bot you want to run locally
- `DISCORD_TOKEN` : The token of the bot you want to run in production
- `DISCORD_DEBUG_CLUSTER_STAGING_ID` : The ID of the "staging" server you want to connect to
- `DISCORD_CLUSTER_STAGING_ID` : The ID of the "production" server you want to connect to
- `GITHUB_TOKEN` : A Github token with permissions to trigger workflows, for now only new branches from [discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager) are tested, since the bot triggers workflows on your behalf
- `DATABASE_URL` : The URL you use to connect to Postgres.

Below is where to find these environment variables:

> [!NOTE]
> For now, you can naively set `DISCORD_DEBUG_TOKEN` and `DISCORD_DEBUG_CLUSTER_STAGING_ID` to the same values as `DISCORD_TOKEN` and `DISCORD_CLUSTER_STAGING_ID` respectively.

- `DISCORD_DEBUG_TOKEN` or `DISCORD_TOKEN`: Found in your bot's page within the [Discord Developer Portal](https://discord.com/developers/applications/):

  <details>
      <summary>Click here for visual.</summary>
      <img width="1440" alt="Screenshot 2024-11-24 at 11 01 19 AM" src="https://github.com/user-attachments/assets/b98bb4e0-8489-4441-83fb-256053aac34d">
  </details>

- `DISCORD_DEBUG_CLUSTER_STAGING_ID` or `DISCORD_CLUSTER_STAGING_ID`: Right-click your staging Discord server and select `Copy Server ID`:

  <details>
      <summary>Click here for visual.</summary>
  <img width="1440" alt="Screenshot 2024-11-24 at 10 58 27 AM" src="https://github.com/user-attachments/assets/0754438c-59ef-4db2-bcaa-c96106c16756">
  </details>

- `GITHUB_TOKEN`: Found in Settings -> Developer Settings (or [here](https://github.com/settings/tokens?type=beta)). Create a new (preferably classic) personal access token with an expiration date to any day less than a year from the current date, and the scopes `repo` and `workflow`.

  <details>
      <summary>Click here for visual.</summary>
      <img width="1440" alt="Screenshot 2024-12-30 at 8 51 59 AM" src="https://github.com/user-attachments/assets/e3467871-bd2c-4f94-b0c5-c8a6ef5ce89e">
  </details>

- `DATABASE_URL`: This contains the connection details for your local database, and has the form `postgresql://user:password@localhost/clusterdev`.

### Verify Setup

Run the following command to run the bot:

```
python src/discord-cluster-manager/bot.py --debug
```

Then in your staging server, use the `/verifyruns` command to test basic functionalities of the bot and the `/verifydb` command to check database connectivity.

> [!NOTE]
> To test functionality of the Modal runner, you also need to be authenticated with Modal. Modal provides free credits to get started.
> To test functionality of the GitHub runner, you may need direct access to this repo which you can ping us for.

## Available Commands

TODO. This is currently a work in progress.

`/run modal <gpu_type>` which you can use to pick a specific gpu, right now defaults to T4

`/run github <NVIDIA/AMD>` which picks one of two workflow files

`/resync` to clear all the commands and resync them

`/ping` to check if the bot is online

## Using the Leaderboard

The main purpose of the Discord bot is to allow servers to host coding competitions through Discord.
The leaderboard was designed for evaluating GPU kernels, but can be adapted easily for other
competitions. The rest of this section will mostly refer to leaderboard submissions in the context
of our GPU Kernel competition.

> [!NOTE]
> All leaderboard commands have the prefix `/leaderboard`, and center around creating, submitting to,
> and viewing leaderboard statistics and information.

### Creating a new Leaderboard

```
/leaderboard create {name: str} {deadline: str} {reference_code: .cu or .py file}
```

The above command creates a leaderboard named `name` that ends at `deadline`. The `reference_code`
has strict function signature requirements, and is required to contain an input generator and a
reference implementation for the desired GPU kernel. We import these functions in our evaluation
scripts for verifying leaderboard submissions and measuring runtime. In the next mini-section, we
discuss the exact requirements for the `reference_code` script.

Each leaderboard `name` can also specify the types of hardware that users can run their kernels on.
For example, a softmax kernel on an RTX 4090 can have different performance characteristics on an
H100. After running the leaderboard creation command, a prompt will pop up where the creator can
specify the available GPUs that the leaderboard evaluates on.

![Leaderboard GPU](assets/img/lb_gpu.png)

#### Reference Code Requirements (Python)

The Discord bot internally contains an `eval.py` script that handles the correctness and timing
analysis for the leaderboard. The `reference_code` that the leaderboard creator submits must have
the following function signatures with their implementations filled out:

```python
# Reference kernel implementation.
def ref_kernel(input: torch.Tensor) -> torch.Tensor:
    # Implement me...

# Generate a list of tensors as input to the kernel
def generate_input() -> List[torch.Tensor]:
    # Implement me...
```

#### Reference Code Requirements (CUDA)

TODO. This is currently a work in progress.

### Submitting to a Leaderboard

```
/leaderboard submit {github / modal} {leaderboard_name: str} {script: .cu or .py file}
```

The leaderboard submission for _Python code_ requires the following function signatures:

```python
# User kernel implementation.
def custom_kernel(input: torch.Tensor) -> torch.Tensor:
    # Implement me...
```

### Other Available Leaderboard Commands

Deleting a leaderboard:

```
/leaderboard delete {name: str}
```

List all active leaderboards and which GPUs they can run on:

```
/leaderboard list
```

List all leaderboard scores (runtime) for a particular leaderboard. (currently deprecated. Doesn't
support multiple GPU types yet)

```
/leaderboard show {name: str}
```

#### GPU Kernel-specific Commands

We plan to add support for the PyTorch profiler and CUDA NSight Compute CLI to allow users to
profile their kernels. These commands are not specific to the leaderboard, but may be helpful for
leaderboard submissions.

## How to add a new GPU to the cluster

If you'd like to donate a GPU to our efforts, we can make you a CI admin in Github and have you add an org level runner https://github.com/organizations/gpu-mode/settings/actions/runners

## Acknowledgements

- Thank you to AMD for sponsoring an MI250 node
- Thank you to NVIDIA for sponsoring an H100 node
- Thank you to Nebius for sponsoring credits and an H100 node
- Thank you Modal for credits and speedy spartup times
- Luca Antiga did something very similar for the NeurIPS LLM efficiency competition, it was great!
- Midjourney was a similar inspiration in terms of UX
