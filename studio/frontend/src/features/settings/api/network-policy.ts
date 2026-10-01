// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type NetworkService = {
  id: string;
  label: string;
  description: string;
};

export type NetworkPolicy = {
  enabled: boolean;
  allowLan: boolean;
  services: Record<string, boolean>;
  catalog: NetworkService[];
};

type ApiNetworkPolicy = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  allow_lan: boolean;
  services: Record<string, boolean>;
  catalog: NetworkService[];
};

function fromApi(policy: ApiNetworkPolicy): NetworkPolicy {
  return {
    enabled: policy.enabled === true,
    allowLan: policy.allow_lan === true,
    services: policy.services ?? {},
    catalog: policy.catalog ?? [],
  };
}

export async function loadNetworkPolicy(): Promise<NetworkPolicy> {
  const res = await authFetch("/api/settings/network-policy");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load network access settings"),
    );
  }
  return fromApi(await res.json());
}

// Writes run one at a time, so the last toggle is also the last write.
let writeQueue: Promise<unknown> = Promise.resolve();

async function putNetworkPolicy(
  policy: Pick<NetworkPolicy, "enabled" | "allowLan" | "services">,
): Promise<NetworkPolicy> {
  const res = await authFetch("/api/settings/network-policy", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      enabled: policy.enabled,
      services: policy.services,
      // biome-ignore lint/style/useNamingConvention: API schema
      allow_lan: policy.allowLan,
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update network access"),
    );
  }
  return fromApi(await res.json());
}

export function updateNetworkPolicy(
  policy: Pick<NetworkPolicy, "enabled" | "allowLan" | "services">,
): Promise<NetworkPolicy> {
  const next = writeQueue
    .catch(() => undefined)
    .then(() => putNetworkPolicy(policy));
  writeQueue = next.catch(() => undefined);
  return next;
}
