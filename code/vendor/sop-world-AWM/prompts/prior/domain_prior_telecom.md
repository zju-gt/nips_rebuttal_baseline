# Domain Prior: Telecom

- Do not stop at the first finding; track every active fault.
- Diagnose in order: airplane mode, data, roaming, network mode, data saver, VPN, APN, SIM, permissions.
- Mobile data: restore base connectivity first, then roaming/network mode/data saver/VPN/refuel, then `run_speed_test`.
- MMS: basic connectivity, network mode >= 3G, APN/MMSC reset, reboot, Wi-Fi calling off, permissions/SIM, then `can_send_mms`.
- If all baseline checks are green but the final check still fails, stop looping and escalate.
