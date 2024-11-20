# discord-cluster-manager

This is the code for the Discord bot we'll be using to queue jobs to a cluster of GPUs that our generous sponsors have provided. Our goal is to be able to queue kernels that can run end to end in seconds that way things feel interactive and social.

The key idea is that we're using Github Actions as a job scheduling engine and primarily making the Discord bot interact with the cluster via issuing Github Actions and and monitoring their status and while we're focused on having a nice user experience on discord.gg/gpumode, we're happy to accept PRs that make it easier for other Discord communities to hook GPUs.

## Why Github Actions

Every triggered job is containerized so we don't have to worry too much about security. We are exploring a K8 like setup but it's just harder to finish in a reasonable timeframe

## Supported schedulers

* GitHub Actions
* Modal
* Slurm (not implemented yet)

## How to run and develop the bot locally

To run and develop the bot locally, you need to add it to your own server. Follow the steps [here](https://discordjs.guide/preparations/setting-up-a-bot-application.html#creating-your-bot) and [here](https://discordjs.guide/preparations/adding-your-bot-to-servers.html#bot-invite-links) to create a bot application and then add it to your server.
Bot needs to be invited using an oauth2 token and needs the `Message Content Intent` permission.
The bot also needs to permissions to read and write messages which is easy to setup if you click on https://discord.com/api/oauth2/authorize?client_id=1303135152091697183&permissions=68608&scope=bot%20applications.commands

After this, you should be able to create a `.env` file with the following environment variables:

- `DISCORD_DEBUG_TOKEN` : The token of the bot you want to run locally
- `DISCORD_DEBUG_CLUSTER_STAGING_ID` : The ID of the staging server you want to connect to
- `GITHUB_TOKEN` : A Github token with permissions to trigger workflows, for now only new branches from [discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager) are tested, since the bot triggers workflows on your behalf

> [!NOTE]
To test functionality of the Modal runner, you also need to be authenticated with Modal. Modal provides free credits to get started.

### How to run the bot

1. Install dependencies with `pip install -r requirements.txt`
2. Create a `.env` file with the environment variables listed above
3. `python src/discord-cluster-manager/bot.py --debug`

### Usage instructions

* `/run modal <gpu_type>` which you can use to pick a specific gpu, right now defaults to T4
* `/run github <NVIDIA/AMD>` which picks one of two workflow files 
* `/resync` to clear all the commands and resync them
* `/ping` to check if the bot is online


## How to test the bot

The smoke test script in `tests/discord-bot-smoke-test.py` should be run to verify basic functionality of the cluster bot. For usage information, run with `python tests/discord-bot-smoke-test.py -h`. Run it against your own server.

[!IMPORTANT]
You need to have multiple environment variables set to run the bot on your own server:

You can run the bot in two modes:
- Production mode: `python discord-bot.py`
- Debug/staging mode: `python discord-bot.py --debug`

When running in debug mode, the bot will use your `DISCORD_DEBUG_TOKEN` and `DISCORD_DEBUG_CLUSTER_STAGING_ID` and display as "Cluster Bot (Staging)" to clearly indicate it's not the production instance.

## How to add a new GPU to the cluster

If you'd like to donate a GPU to our efforts, we can make you a CI admin in Github and have you add an org level runner https://github.com/organizations/gpu-mode/settings/actions/runners

## Acknowledgements

* Thank you to AMD for sponsoring an MI250 node
* Thank you to NVIDIA for sponsoring an H100 node
* Thank you to Nebius for sponsoring credits and an H100 node
* Thank you Modal for credits and speedy spartup times
* Luca Antiga did something very similar for the NeurIPS LLM efficiency competition, it was great!
* Midjourney was a similar inspiration in terms of UX
