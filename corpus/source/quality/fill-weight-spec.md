# Fill Weight and Volume Specification with SPC Limits

## Purpose and Scope

This specification sets target fill weights, tolerances, and statistical process control (SPC) limits for products filled at Riverbend Packaging. It defines how operators check fills and how Quality responds to out-of-control conditions. It covers all filler heads on Line 1.

## Fill Targets

Fill is controlled by net weight, converted from volume using the product density.

| Product | Container | Target net weight | Lower spec | Upper spec |
|---|---|---|---|---|
| Product A | 500 mL | 505.0 g | 500.0 g | 510.0 g |
| Product A | 250 mL | 252.5 g | 249.0 g | 256.0 g |
| Product A | 1 L | 1010.0 g | 1002.0 g | 1018.0 g |

The lower specification limit is a legal minimum. Under-filled packages may never be shipped.

## Sampling

| Condition | Sample | Frequency |
|---|---|---|
| Startup | 1 bottle from each head | Before release |
| Normal run | 5 consecutive bottles | Every 15 minutes |
| After changeover | 10 bottles | Before release |
| After a filler stop over 10 minutes | 5 bottles | At restart |

Weigh bottles on a calibrated check scale, subtracting the average tare weight from the empty container study.

## SPC Limits

Control charts use X-bar and R charts based on subgroups of five.

| Chart | Center line | Upper control limit | Lower control limit |
|---|---|---|---|
| X-bar, 500 mL | 505.2 g | 507.0 g | 503.4 g |
| Range, 500 mL | 2.1 g | 4.4 g | 0 g |

Control limits are recalculated quarterly from at least 25 in-control subgroups. Specification limits are never used as control limits.

## Out-of-Control Rules

1. Any point outside the control limits.
2. Seven consecutive points on one side of the center line.
3. Six points in a row steadily rising or falling.
4. Two of three points near a control limit.

When any rule triggers:

1. Take an additional sample of five bottles.
2. Notify the line lead and check filler head settings.
3. If a point is outside a specification limit, stop the line and notify the Quality Engineer.
4. Hold all product filled since the last in-control check.

## Filler Head Adjustments

Operators may adjust the fill volume dial by no more than one division per adjustment and must reweigh before the next adjustment. Adjustments larger than that require a maintenance technician. Record every adjustment in the SPC log.

## Capability

Process capability is evaluated monthly. The minimum acceptable Cpk is 1.33. If Cpk falls below that value, Quality opens a corrective action and maintenance reviews filler head wear.

## Check Scale Verification

The check scale is verified at the start of each shift with a certified 500 g test weight. The tolerance is plus or minus 0.1 g. A scale that fails verification is removed from use and all weights since the last good check are reevaluated.

## Pump Influence

Fill variation can result from supply pressure changes at the filler bowl. When fill variation rises without head-level explanations, notify maintenance to check the RP-200 pump discharge pressure and bowl level control.

## Records

Weights are logged in the SPC system. Charts are reviewed daily by the line lead and weekly by the Quality Engineer. Records are kept for two years.

## Shift Handoff and Reporting

At the end of each shift the line lead reviews the control charts with the incoming lead and notes any open adjustments, held product, or suspected equipment issues in the shift log. Operators must not leave a chart with an unresolved out-of-control signal without handing it to a named person. Quality summarizes fill performance weekly, including the number of out-of-control signals, the average giveaway above target, and any holds. Reducing giveaway while staying above the legal minimum is a continuous improvement goal, but no target may be pursued by lowering the filler setting without Quality approval.
