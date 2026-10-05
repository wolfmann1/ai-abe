# Runbook: Replica lag

Applies to: read replicas of the artifact cache service (fictional example service "Stash").

## Symptoms

Remote offices report that new artifacts take minutes to appear, or builds fetch an older version than the one just published. The replica lag alert fires when a replica is more than 300 seconds behind the primary.

## First checks

Check lag for every replica with `stash-admin replicas status`. A single lagging replica points to its network link or its own disk; every replica lagging at once points to the primary.

On the lagging replica, check the replication journal with `stash-admin journal --tail 50`. Repeated "retry" lines with the same journal number mean one large file is failing to transfer.

## Fix

For a single lagging replica, restart its replication worker with `systemctl restart stash-replicator`. Lag should start falling within five minutes.

If the journal shows one file retrying, skip it with `stash-admin journal --skip <number>` and schedule a full verify of that replica overnight with `stash-admin verify --full`.

If every replica is lagging, do not restart replicas. Escalate to the primary on-call engineer, because the primary is either overloaded or its journal disk is full.

## After the fix

Record the peak lag, the cause and the fix in the incident log. Lag above 30 minutes during working hours needs a short post-incident review.
