# Licensed video ingestion for assisted review

The product must distinguish access to a video from permission to use that video for
machine-learning training. An integration is eligible only when the customer owns the
recording or its contract explicitly grants download, processing, derived-label, and
commercial model-training rights.

## Recommended integration order

1. Local files recorded or licensed by the customer. This remains the default and
   requires no cloud dependency.
2. Pixellot Partner API. Its official documentation exposes event webhooks and a
   `convertToMP4` operation that produces a download link. Production credentials and
   contractual ML-training rights must be obtained from Pixellot and the customer.
3. Customer-owned Veo, Spiideo, or Hudl recordings. Their documented products support
   full-match or custom downloads, but a public unattended full-match download API was
   not confirmed. Treat these as manual imports until a vendor contract supplies an API.
4. Wyscout only under a negotiated agreement that explicitly permits ML training.
   Standard platform access or the Data API is not sufficient evidence of those rights.

YouTube is not an ingestion source. YouTube API policies prohibit downloading,
importing, caching, or storing audiovisual content without prior written approval.

## Proposed automated workflow

1. Receive a webhook for a completed customer-owned recording.
2. Verify tenant, recording ownership, consent, and the stored license policy.
3. Request an MP4 export and download it into local encrypted temporary storage.
4. Run the research master only to propose candidate review blocks.
5. Present candidates, hard negatives, and uncovered timeline blocks to a person.
6. Persist the corrected human labels with provenance and reviewer identity.
7. Delete the downloaded source according to the customer's retention policy.
8. Admit only verified, commercially eligible labels to commercial training.

## Vendor references

- Pixellot Partner API and onboarding: https://docs.pixellot.tv/portal/en/kb/articles/api-reference-14-9-2022
- Pixellot webhooks and MP4 conversion: https://docs.pixellot.tv/portal/en/kb/articles/api-webhook-subscriptions-8-11-2023
- Veo plan download capabilities: https://support.veo.co/hc/en-us/articles/4450464075025-Veo-subscription-plans
- Spiideo custom downloads: https://support.spiideo.com/en/articles/8661961-create-a-custom-download
- Hudl Wyscout API offering: https://www.hudl.com/products/wyscout/football-api
- YouTube API developer policies: https://developers.google.com/youtube/terms/developer-policies
