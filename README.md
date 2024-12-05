# discord-cluster-manager

This is the code for the Discord bot we'll be using to queue jobs to a cluster of GPUs that our generous sponsors have provided. Our goal is to be able to queue kernels that can run end to end in seconds that way things feel interactive and social.

The key idea is that we're using Github Actions as a job scheduling engine and primarily making the Discord bot interact with the cluster via issuing Github Actions and and monitoring their status and while we're focused on having a nice user experience on discord.gg/gpumode, we're happy to accept PRs that make it easier for other Discord communities to hook GPUs.

[Demo!](https://www.youtube.com/watch?v=-u7kX_vpLfk)

## Supported schedulers

* GitHub Actions
* Modal
* Slurm (not implemented yet)

## How to run and develop the bot locally

To run and develop the bot locally, you need to add it to your own server. Follow the steps [here](https://discordjs.guide/preparations/setting-up-a-bot-application.html#creating-your-bot) and [here](https://discordjs.guide/preparations/adding-your-bot-to-servers.html#bot-invite-links) to create a bot application and then add it to your server.

Here is a visual walk-through of the steps (after clicking on the New Application button):

- The bot needs the `Message Content Intent` permission.
  <details>
    <summary>Click here for visual.</summary>
    <img width="1440" alt="Screenshot 2024-11-24 at 10 44 46 AM" src="https://github.com/user-attachments/assets/7c873a9d-55b8-4aea-8c9a-9d2909405f03">
  </details>

- The bot also needs `applications.commands` and `bot` scopes.

  <details>
      <summary>Click here for visual.</summary>
    <img width="1440" alt="Screenshot 2024-11-24 at 12 34 09 PM" src="https://github.com/user-attachments/assets/31302214-1d5a-416a-b7b4-93a44442be51">
  </details>

- The bot also needs to permissions to read and write messages which is easy to setup if you click on [this link](https://discord.com/api/oauth2/authorize?client_id=1303135152091697183&permissions=68608&scope=bot%20applications.commands). 
Finally, generate an invite link for the bot and enter it into any browser.

  <details>
      <summary>Click here for visual.</summary>
      <img width="1440" alt="Screenshot 2024-11-24 at 12 44 08 PM" src="https://github.com/user-attachments/assets/54c34b6b-c944-4ce7-96dd-e40cfe79ffb3">
  </details>


> [!NOTE]
> Bot permissions involving threads/mentions/messages should suffice, but you can naively give it `Administrator` since it's just a test bot in your own testing Discord server.  

### Environment Variables
After this, you should be able to create a `.env` file with the following environment variables:

- `DISCORD_DEBUG_TOKEN` : The token of the bot you want to run locally
- `DISCORD_DEBUG_CLUSTER_STAGING_ID` : The ID of the staging server you want to connect to
- `GITHUB_TOKEN` : A Github token with permissions to trigger workflows, for now only new branches from [discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager) are tested, since the bot triggers workflows on your behalf


Below is where to find these environment variables:
- **`DISCORD_DEBUG_TOKEN` or `DISCORD_TOKEN`**: Found in your bot's page within the [Discord Developer Portal](https://discord.com/developers/applications/):

  <details>
      <summary>Click here for visual.</summary>
      <img width="1440" alt="Screenshot 2024-11-24 at 11 01 19 AM" src="https://github.com/user-attachments/assets/b98bb4e0-8489-4441-83fb-256053aac34d">
  </details>
  
- **`DISCORD_DEBUG_CLUSTER_STAGING_ID` or `DISCORD_CLUSTER_STAGING_ID`**: Right-click your staging Discord server and select `Copy Server ID`:

  <details>
      <summary>Click here for visual.</summary>
  <img width="1440" alt="Screenshot 2024-11-24 at 10 58 27 AM" src="https://github.com/user-attachments/assets/0754438c-59ef-4db2-bcaa-c96106c16756">
  </details>
  
- **`GITHUB_TOKEN`**: Found in Settings -> Developer Settings (or [here](https://github.com/settings/tokens?type=beta)).

### How to run the bot

1. Install dependencies with `pip install -r requirements.txt`
2. Create a `.env` file with the environment variables listed above
3. `python src/discord-cluster-manager/bot.py --debug`

### Usage instructions

> [!NOTE]
> To test functionality of the Modal runner, you also need to be authenticated with Modal. Modal provides free credits to get started.
> 
> To test functionality of the GitHub runner, you may need direct access to this repo.

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
