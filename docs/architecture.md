# EXO Architecture overview

EXO uses an _Event Sourcing_ architecture, and Erlang-style _message passing_. To facilitate this, we've written a channel library extending anyio channels with inspiration from tokio::sync::mpsc. 

Each logical module - designed to be functional independently of the others - communicates with the rest of the system by sending messages on topics.

## Systems

There are currently 5 major systems:

- Master
    
    Executes placement and orders events through a single writer

- Worker
    
    Schedules work on a node, gathers system information, etc.#

- Runner
    
    Executes inference jobs (for now) in an isolated process from the worker for fault-tolerance.

- API
    
    Runs a python webserver for exposing state and commands to client applications

- Election
    
    Implements a distributed algorithm for master election in unstable networking conditions

## Topics

There are currently 5 topics:

- Commands

    The API and Worker instruct the master when the event log isn't sufficient. Namely placement and catchup requests go through Commands atm.

- Local Events

    All nodes write events here, the master reads those events and orders them

- Global Events

    The master writes events here, all nodes read from this topic and fold the produced events into their `State`

- Election Messages

    Before establishing a cluster, nodes communicate here to negotiate a master node.

- Connection Messages

    The networking system write mdns-discovered hardware connections here.


## Event Sourcing

Lots has been written about event sourcing, but it lets us centralize faulty connections and message ACKing with the following model.

Whenever a device produces side effects, it captures those side effects in an `Event`. `Event`s are then "applied" to their model of `State`, which is globally distributed across the cluster. Whenever a command is received, it is combined with state to produce side effects, captured in yet more events. The rule of thumb is "`Event`s are past tense, `Command`s are imperative". Telling a node to perform some action like "place this model" or "Give me a copy of the event log" is represented by a command (The worker's `Task`s are also commands), while "this node is using 300GB of ram" is an event. Notably, `Event`s SHOULD never cause side effects on their own. There are a few exceptions to this, we're working out the specifics of generalizing the distributed event sourcing model to make it better suit our needs

## Purity

A significant goal of the current design is to make data flow explicit. Classes should either represent simple data (`CamelCaseModel`s typically, and `TaggedModel`s for unions) or active `System`s (Erlang `Actor`s), with all transformations of that data being "referentially transparent" - destructure and construct new data, don't mutate in place. We have had varying degrees of success with this, and are still exploring where purity makes sense.
