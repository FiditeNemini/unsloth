// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import {
  type NetworkPolicy,
  loadNetworkPolicy,
  updateNetworkPolicy,
} from "../api/network-policy";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

/** Offline unless the owner turns access on: one master switch, then each service. */
export function NetworkAccessSection() {
  const isOwner = useIsAccountOwner();
  const [policy, setPolicy] = useState<NetworkPolicy | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadNetworkPolicy()
      .then(setPolicy)
      .catch((error: unknown) =>
        toast.error(error instanceof Error ? error.message : String(error)),
      );
  }, []);

  const readOnly = !isOwner || policy === null || saving;

  // Never rejects: failures are reported here and the switch reverts.
  const save = async (next: NetworkPolicy) => {
    const previous = policy;
    setPolicy(next); // optimistic; reverted below on failure
    setSaving(true);
    try {
      setPolicy(await updateNetworkPolicy(next));
    } catch (error: unknown) {
      setPolicy(previous);
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <SettingsSection
      title="Network access"
      description={
        <>
          Studio is offline by default: nothing leaves this Mac unless it is
          allowed here. Changes apply straight away to Studio and to models,
          training and exports started afterwards; ones already running keep
          their previous setting.
          {isOwner ? null : " Only the owner can change this."}
        </>
      }
    >
      <SettingsRow
        label="Allow network access"
        description={
          policy?.enabled
            ? "On: only the services switched on below can connect."
            : "Off: every outside connection is blocked, including by libraries."
        }
      >
        <Switch
          aria-label="Allow network access"
          checked={policy?.enabled ?? false}
          disabled={readOnly}
          onCheckedChange={(enabled) => {
            if (policy) {
              save({ ...policy, enabled });
            }
          }}
        />
      </SettingsRow>

      {policy?.catalog.map((service) => (
        <SettingsRow
          key={service.id}
          label={service.label}
          description={service.description}
          className={cn(!policy.enabled && "opacity-60")}
        >
          <Switch
            aria-label={service.label}
            checked={policy.services[service.id] ?? false}
            disabled={readOnly || !policy.enabled}
            onCheckedChange={(on) => {
              save({
                ...policy,
                services: { ...policy.services, [service.id]: on },
              });
            }}
          />
        </SettingsRow>
      ))}

      <SettingsRow
        label="Local network"
        description="Allow addresses on your local network (192.168.x.x, 10.x.x.x) even while network access is off, for servers you run yourself."
      >
        <Switch
          aria-label="Allow local network"
          checked={policy?.allowLan ?? false}
          disabled={readOnly}
          onCheckedChange={(allowLan) => {
            if (policy) {
              save({ ...policy, allowLan });
            }
          }}
        />
      </SettingsRow>
    </SettingsSection>
  );
}
