## Description

Please provide a brief summary of the changes in this pull request.

## Checklist

Before submitting this PR, ensure the following steps have been completed:

- [ ] Run the slash command `/verifyruns` on your own server.
  - Run the cluster bot on your server:
    ```bash
    python discord-bot.py
    ```
  - Start training runs with the slash command `/verifyruns`.
  - Verify that the bot eventually responds with:
    ```
    ✅ All runs completed successfully!
    ```
    (It may take a few minutes for all runs to finish. In particular, the GitHub
    runs may take a little longer. The Modal run is typically quick.)
  For more information on running a cluster bot on your own server, see
  README.md.
