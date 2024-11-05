# discord-cluster-manager

This is the code for the Discord bot we'll be using to queue jobs to a cluster of GPUs that our generous sponsors have provided.

The key idea is that we're using Github Actions as a job scheduling engine and primarily making the Discord bot interact with the cluster via issuing Github Actions and and monitoring their status and while we're focused on having a nice user experience on discord.gg/gpumode, we're happy to accept PRs that make it easier for other Discord communities to hook GPUs.

## How to run the bot locally

1. Install dependencies with `pip install -r requirements.txt`
2. Create a `.env` file
3. `python bot.py`

Right now the bot is running on my macbook but will some more permanent location

## Why Github Actions

Every triggered job is containerized so we don't have to worry too much about security. We are exploring a K8 like setup but it's just harder to finish in a reasonable timeframe

### How to test the bot

Instead of testing on GPU MODE directly we can leverage a staging environment called "Discord Cluster Staging". If you need access to this server please ping "Seraphim"

Bot needs to be invited using an oauth2 token and needs the `Message Content Intent` permission

The bot also needs to permissions to read and write messages which is easy to setup if you click on https://discord.com/api/oauth2/authorize?client_id=1303135152091697183&permissions=68608&scope=bot%20applications.commands

### How to add a new GPU to the cluster

Github has some nice instructions here https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners but essentially the whole thing works by running a script on some GPU people own.

### Future work
* Maybe we shouldn't use Github Action and can roll our own thing?
* Make registering new GPUs simpler

## Acknowledgements
* Luca Antiga did something very similar for the NeurIPS LLM efficiency competition, it was great!
* Midjourney was a similar inspiration in terms of UX