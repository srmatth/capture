# systemd units

User-mode units for the capture pipeline. Install after `/srv/data/capture/`
exists and the app is running.

```bash
mkdir -p ~/.config/systemd/user
cp systemd/*.service systemd/*.path ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now \
  capture-transcribe.path \
  capture-classify.path \
  capture-embed.path
```

Notes:

- Only the `.path` units are enabled directly. They fire the corresponding
  `.service` on filesystem change; the service files themselves don't need
  `[Install]` sections.
- Each service reads `~/.config/matthewshome/capture.env` for the
  Anthropic API key. If that env-file is missing you'll see a `status=203/EXEC`
  or the worker will crash on first Claude call.
- `OnFailure=notify-fail@%n.service` piggybacks on the shared notify template
  set up in Phase 0. Every worker failure pings the `alerts` ntfy topic.
- The classify and embed services will `NotImplementedError` on first fire
  until Steps 9 and 10 are built — that's expected. Leave them enabled so
  the plumbing is proven before the workers exist.

Verify:
```bash
systemctl --user list-timers | grep capture     # (path units aren't timers, but list-units shows them)
systemctl --user list-units --state=active | grep capture
```
