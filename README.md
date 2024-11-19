# discord-cluster-manager

This is the code for the Discord bot we'll be using to queue jobs to a cluster of GPUs that our generous sponsors have provided.

The key idea is that we're using Github Actions as a job scheduling engine and primarily making the Discord bot interact with the cluster via issuing Github Actions and and monitoring their status and while we're focused on having a nice user experience on discord.gg/gpumode, we're happy to accept PRs that make it easier for other Discord communities to hook GPUs.

## How to run the bot locally

1. Install dependencies with `pip install -r requirements.txt`
2. Create a `.env` file
3. `python discord-bot.py`
4. In the staging channel @Cluster-bot with a sample `train.py`

Right now the bot is running on my macbook but will some more permanent location

## Supported schedulers

* GitHub Actions
* Modal
* Slurm (not implemented yet)

## Usage instructions

`@Cluster-bot NVIDIA/AMD/MODAL` depending on which scheduleer you want to use. MODAL is configured by default to use T4 because that's cheap but it works with any GPU

## Why Github Actions

Every triggered job is containerized so we don't have to worry too much about security. We are exploring a K8 like setup but it's just harder to finish in a reasonable timeframe

### How to test the bot

Instead of testing on GPU MODE directly we can leverage a staging environment called "Discord Cluster Staging". If you need access to this server please ping "Seraphim", however, you can also test the bot on your own server by following the instructions below.

### How to add the bot to a personal server

For testing purposes, bot can be run on a personal server as well. Follow the steps [here](https://discordjs.guide/preparations/setting-up-a-bot-application.html#creating-your-bot) and [here](https://discordjs.guide/preparations/adding-your-bot-to-servers.html#bot-invite-links) to create a bot application and then add it to your server.
After doing that, you can add a new environment variable called `DISCORD_DEBUG_TOKEN` to your `.env` file and set it to the bot token you got from the Discord Developer Portal.

You can run the bot in two modes:
- Production mode: `python discord-bot.py`
- Debug/staging mode: `python discord-bot.py --debug`

When running in debug mode, the bot will use your `DISCORD_DEBUG_TOKEN` and display as "Cluster Bot (Staging)" to clearly indicate it's not the production instance.

Bot needs to be invited using an oauth2 token and needs the `Message Content Intent` permission.

The bot also needs to permissions to read and write messages which is easy to setup if you click on https://discord.com/api/oauth2/authorize?client_id=1303135152091697183&permissions=68608&scope=bot%20applications.commands

### How to add a new GPU to the cluster

If you'd like to donate a GPU to our efforts, we can make you a CI admin in Github and have you add an org level runner https://github.com/organizations/gpu-mode/settings/actions/runners


## Acknowledgements

* Thank you to AMD for sponsoring an MI250 node
* Thank you to NVIDIA for sponsoring an H100 node
* Thank you to Nebius for sponsoring credits and an H100 node
* Thank you Modal for credits and speedy spartup times
* Luca Antiga did something very similar for the NeurIPS LLM efficiency competition, it was great!
* Midjourney was a similar inspiration in terms of UX
