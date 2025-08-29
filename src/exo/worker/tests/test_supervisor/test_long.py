import asyncio
from logging import Logger
from typing import Callable

import pytest

from exo.shared.logging import logger_test_install
from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.openai_compat import FinishReason
from exo.shared.types.common import Host
from exo.shared.types.events.chunks import TokenChunk
from exo.shared.types.tasks import (
    Task,
    TaskId,
)
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.runner.runner_supervisor import RunnerSupervisor


@pytest.fixture
def user_message():
    """Override the default message to ask about France's capital"""
    return "What is the capital of France?"

@pytest.fixture
def lorem_ipsum() -> str:
    return """        
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Phasellus rhoncus felis in velit tempus tristique. Nullam ipsum lectus, tristique a eros quis, ullamcorper accumsan lorem. Aliquam ut auctor elit, finibus porttitor neque. In cursus augue facilisis ante ullamcorper, at sollicitudin quam aliquam. Etiam ac lacinia lacus, et aliquet nunc. Phasellus nisi ex, feugiat quis dolor non, mollis consequat nulla. Suspendisse gravida, sem non lobortis viverra, turpis lacus elementum orci, in tristique augue tortor nec mauris. Curabitur aliquet lorem in rhoncus mollis. Aliquam pulvinar elit odio, ac feugiat magna luctus nec. Pellentesque non risus egestas, pellentesque arcu tincidunt, gravida risus. Etiam ut lorem ac lorem pharetra efficitur. Donec augue arcu, varius nec lorem vitae, suscipit semper tellus. Aliquam dignissim quis augue id fermentum. Proin aliquet pellentesque est, eget tincidunt odio ullamcorper vel. Suspendisse potenti.
Aenean imperdiet justo sit amet erat aliquet tristique. Sed tempus, turpis a cursus lobortis, ante sem imperdiet est, eu dapibus sapien velit eget elit. Donec feugiat sed risus sed scelerisque. Donec posuere tempor orci, sit amet pellentesque est efficitur non. Vivamus sodales pretium purus, sed rutrum enim auctor ut. Cras pharetra vitae libero et hendrerit. Sed nec tempus odio. Proin blandit facilisis scelerisque. Nulla in mattis mi. Etiam bibendum efficitur aliquam. Proin ut risus aliquet, rhoncus lectus non, rhoncus arcu. Nam nibh felis, ultrices a elit sed, ultricies sollicitudin tellus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Maecenas faucibus magna ut purus imperdiet faucibus. Nam fermentum nulla fermentum magna aliquam, vel lacinia neque euismod. Donec tincidunt sed neque non facilisis.
Proin id lorem cursus, vehicula ante non, lacinia metus. Nam egestas dui a iaculis convallis. Ut suscipit justo est, nec pharetra ante accumsan ac. Pellentesque nec nisi ipsum. Duis non arcu neque. Curabitur non luctus purus. Phasellus pulvinar commodo lacus sit amet auctor. Ut ut mattis metus, eu auctor arcu. Etiam a suscipit est. Morbi orci mauris, suscipit tempus fermentum vel, luctus faucibus lectus. Aliquam a euismod arcu. Suspendisse porttitor eget libero vitae laoreet.
Fusce congue lorem mi, a mollis felis efficitur quis. Quisque lobortis scelerisque arcu, a varius sapien. Nulla eget orci non urna imperdiet tincidunt. Nunc mi massa, consectetur id lorem consectetur, molestie dignissim sem. Suspendisse et augue magna. Mauris id tempus velit, cursus suscipit tortor. Duis non mi non nisi fringilla maximus in et erat.
Proin consequat sapien eget tellus aliquam ultrices. Nunc hendrerit semper massa, pulvinar sodales ipsum condimentum eu. Proin vel ligula venenatis, lobortis lectus eu, vehicula justo. Mauris eu arcu at orci vehicula feugiat non eu metus. Duis ut vestibulum quam. Maecenas dolor elit, egestas ut purus sit amet, convallis lobortis massa. Ut volutpat augue ac ante consectetur dignissim. Maecenas vitae felis elementum, semper augue eu, auctor dolor. Ut pulvinar convallis tortor non volutpat. Curabitur vulputate sem sodales sapien pretium ultrices. Sed luctus libero vitae urna eleifend tincidunt. Proin pulvinar imperdiet cursus. Suspendisse ullamcorper laoreet leo dapibus tincidunt. Pellentesque molestie elementum felis.
Integer vitae congue nulla. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Vestibulum elit velit, malesuada quis ipsum et, imperdiet varius velit. Nam tristique viverra maximus. Curabitur eget semper lectus. Sed vitae lorem sit amet mi lacinia posuere ac a risus. Pellentesque et magna nisl. In hac habitasse platea dictumst. Aenean suscipit, nibh vitae sollicitudin commodo, risus mi commodo neque, nec venenatis velit augue sed massa. Nam tempus, arcu id eleifend auctor, est dui viverra odio, vel convallis arcu dolor id quam. Ut malesuada ligula vel interdum eleifend. In posuere ultrices tincidunt. Sed non enim sit amet lectus sagittis mattis eu at sapien. Pellentesque eu urna mollis, vehicula dolor eget, lobortis nisl. Suspendisse ex nisi, iaculis non sapien ac, fringilla rutrum dolor. Quisque pretium mauris nec ante gravida, sed laoreet neque viverra.
Donec mattis orci sit amet tincidunt maximus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Curabitur tristique venenatis lectus, vel pulvinar sem. Sed vel dolor lacinia, aliquet nisi ac, bibendum libero. Nullam vulputate euismod augue ac imperdiet. Proin at fermentum sapien. Nam et fringilla lorem. Aenean sed lacus sed tellus sodales mattis ut rutrum ex. Nulla ligula diam, interdum quis faucibus sit amet, laoreet vel massa. Fusce mauris massa, tempor quis tempus nec, dictum a ligula. Ut at dapibus sapien. Nullam sem lorem, sollicitudin non dui a, consequat molestie mauris. Quisque sem nulla, vehicula nec vulputate ac, viverra in massa. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur pretium venenatis nisi non bibendum. Nam vitae ligula auctor, rutrum lectus eget, feugiat augue.
Ut nunc risus, vehicula at metus non, consequat suscipit risus. Mauris eget sem in neque tincidunt iaculis. Pellentesque lacus leo, molestie ut pharetra sit amet, porta nec neque. Aliquam eu bibendum odio. Proin tempus bibendum ornare. Morbi non risus vitae ante tempor porta quis sed augue. Nullam hendrerit nulla in eleifend tincidunt. Integer suscipit ligula at nunc blandit vehicula. Nam porttitor leo in turpis suscipit malesuada. Etiam sodales nunc nisi, pharetra malesuada nibh varius in. Cras quis pellentesque augue, vitae convallis velit. In et dui lorem. Integer semper eros eget augue posuere, ac elementum tellus convallis. Praesent blandit tempus ultrices. Suspendisse nec dui vitae neque varius eleifend. Sed pretium metus leo, id viverra tellus scelerisque in.
Aenean sodales urna vitae lobortis cursus. Sed vitae pellentesque erat, fermentum pellentesque urna. Suspendisse potenti. Sed porttitor placerat turpis non vestibulum. Duis in nisi non purus venenatis tempus non eu nisi. Sed bibendum sapien vitae ultricies condimentum. Integer vel mattis lectus, consequat congue ex. Cras convallis odio volutpat nulla vehicula efficitur. Pellentesque eget justo neque. Morbi mattis vitae magna et suscipit. Etiam orci sapien, tincidunt non tellus eget, laoreet vestibulum massa. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Mauris nec nisi enim. Donec risus odio, lobortis in odio malesuada, laoreet rutrum urna. Nunc sit amet euismod quam.
Fusce rhoncus ullamcorper nunc, ut pellentesque nisi dictum sed. Fusce sem mi, bibendum ut dictum at, porta in libero. Pellentesque placerat mollis sapien, sed eleifend lorem consequat in. Phasellus vel tempor ligula. Pellentesque tincidunt suscipit tortor vel blandit. Maecenas purus mi, mattis ac aliquam vel, rutrum eu nulla. Proin rhoncus nec sem a congue. Pellentesque sit amet sapien quam. Sed hendrerit neque id venenatis dignissim.
Vestibulum laoreet eu felis nec aliquam. Praesent gravida ornare odio nec porttitor. Donec ut tellus eros. Proin fringilla urna augue, vitae ornare leo varius non. Curabitur consectetur, purus in iaculis finibus, lectus lacus porttitor dolor, nec eleifend tellus massa eget tellus. Mauris sit amet convallis risus, a fermentum lorem. Suspendisse potenti. Curabitur vulputate finibus maximus. Interdum et malesuada fames ac ante ipsum primis in faucibus. In vel erat pellentesque, rhoncus magna vel, scelerisque mauris.
Nulla facilisi. Morbi mattis felis nec accumsan varius. Vestibulum in sodales arcu. Vivamus egestas, ante nec dapibus vestibulum, tellus ipsum rhoncus mi, at fermentum sapien justo nec turpis. Quisque rhoncus, urna sit amet imperdiet cursus, tortor lacus ultricies sapien, eu bibendum ligula enim id mi. Sed sem leo, pharetra in pulvinar sed, faucibus sed dui. Morbi tempus erat nec neque placerat tincidunt.
Quisque ut lorem sodales magna faucibus mattis. Aenean dui neque, gravida ut fringilla non, fermentum sit amet dolor. Mauris a sapien lacinia, elementum dolor in, sagittis metus. Donec viverra magna non lorem rutrum, at eleifend lacus volutpat. Nunc sit amet dolor tempor, blandit sapien a, consectetur magna. Suspendisse maximus nunc nec imperdiet aliquet. Nunc aliquam interdum purus quis pretium. Mauris molestie feugiat pellentesque. Nunc maximus, est sed consequat malesuada, risus turpis consequat velit, ac feugiat nunc magna vitae ligula. Vestibulum tincidunt massa ante, vitae pellentesque tortor rutrum sed. Aliquam vel est libero. Suspendisse et convallis orci. Cras sed lorem consectetur, blandit massa sit amet, semper neque. Vestibulum et mi euismod, imperdiet justo non, facilisis libero.
Sed at lacus ac tortor dictum tempus. Integer commodo purus lacus, ut pretium est tempor ac. Ut vulputate nulla magna, ac facilisis velit commodo in. Interdum et malesuada fames ac ante ipsum primis in faucibus. Donec pellentesque congue nibh nec eleifend. Ut ante turpis, sodales sed aliquet quis, tempus eu dui. Proin et eros non risus porttitor pharetra.
Mauris a urna id justo gravida ultrices. Mauris commodo sed ipsum a dictum. In posuere luctus scelerisque. Morbi sit amet gravida ipsum. Quisque vel dui sit amet ex lobortis eleifend non vel neque. Fusce sit amet imperdiet felis, eu tempor diam. Pellentesque sit amet turpis in libero tristique posuere. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Mauris quis est suscipit, tristique odio elementum, molestie nibh. Maecenas ex dui, pulvinar quis pellentesque sed, imperdiet nec mauris. Pellentesque ultrices at mauris eget fringilla. Donec bibendum rhoncus felis, ut pretium nulla eleifend commodo.
Ut euismod erat accumsan tincidunt sagittis. Proin eget massa ex. Suspendisse at faucibus enim, vitae posuere mi. Cras nec ex finibus, porttitor purus quis, efficitur libero. Nulla sagittis ornare iaculis. Donec venenatis dui ut libero aliquam lobortis. Vestibulum imperdiet lorem urna, eget gravida orci sollicitudin ut. Quisque ultrices tortor at quam laoreet aliquet. Pellentesque tincidunt consequat pharetra. Cras a lacinia erat. Mauris sed neque lobortis ipsum facilisis hendrerit.
Cras at orci odio. Curabitur eros metus, consequat non placerat et, tincidunt at turpis. Morbi quis viverra metus. Vestibulum molestie, ex at suscipit finibus, ex magna pellentesque nisi, eu ullamcorper nisl sapien eu quam. Phasellus volutpat lacinia enim, nec fermentum augue tincidunt ut. Duis rutrum purus eu nulla elementum, a faucibus odio fringilla. Sed cursus risus neque, dictum luctus tortor tempus eu.
Mauris non arcu eu nunc faucibus tincidunt id quis dolor. Quisque ac fringilla libero. Sed non ligula ut nunc auctor consequat vitae eget metus. Ut suscipit leo quam, vitae ultrices urna feugiat eu. Vestibulum volutpat nisl quis nunc pretium, vel viverra orci fringilla. Proin erat nibh, laoreet nec nisi sit amet, volutpat efficitur nunc. Cras id tortor quis lectus imperdiet rutrum non id purus. Proin efficitur ligula non dapibus consectetur. Nam quis quam eget dui facilisis scelerisque. Praesent non bibendum risus. Etiam imperdiet nisi id consectetur porta. In pretium nulla ut leo ultricies rhoncus.
Curabitur non vehicula purus. Cras et justo risus. Duis et rutrum urna. Aliquam condimentum purus nec ante dignissim rhoncus. Vestibulum commodo pharetra eros, ac euismod orci rutrum vel. Integer sed cursus erat, euismod accumsan libero. Nullam ut odio sit amet nibh tempor congue. Pellentesque porttitor aliquam ipsum, sit amet facilisis quam fringilla ac. Aliquam scelerisque tempor nisl in tempor. Sed vestibulum, tellus sit amet mattis pellentesque, eros diam convallis felis, id pellentesque massa leo quis dolor. Integer dignissim orci lorem, vel porttitor felis blandit et. Nam ultrices enim sed elementum accumsan. Fusce rutrum, quam et feugiat maximus, lorem leo porttitor ex, a eleifend risus odio consectetur lacus. In hac habitasse platea dictumst. Aenean pharetra erat tellus, at tempus urna iaculis ut. Ut ac mi eu lorem volutpat egestas.
Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Praesent porttitor tempor ligula. Quisque mollis arcu in metus ornare pellentesque. Aenean ultrices mollis quam quis sodales. Maecenas a cursus elit, id gravida tortor. Donec vel purus magna. Aliquam elementum est sed convallis fermentum. Nam nec eros arcu. Pellentesque sed eros a lacus sagittis maximus. Integer et tellus id libero dapibus convallis. Maecenas viverra, purus facilisis porttitor tincidunt, tellus lacus elementum dui, sed porttitor sem justo a lorem. Curabitur ipsum odio, efficitur quis efficitur at, tempus aliquet nisi. Aliquam ultrices tortor in arcu vulputate, vel iaculis lorem facilisis. Cras eleifend laoreet feugiat. Integer placerat blandit sem, mattis elementum purus pellentesque quis. Etiam vel arcu ut mi commodo placerat non id tortor.
"""

@pytest.mark.asyncio
async def test_supervisor_long_prompt_response(
    pipeline_shard_meta: Callable[..., PipelineShardMetadata],
    hosts: Callable[..., list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
    lorem_ipsum: str,
    logger: Logger,
):
    """Test that asking for the capital of France returns 'Paris' in the response"""
    logger_test_install(logger)

    model_meta = MODEL_CARDS['llama-3.2-1b'].metadata
    model_shard_meta = PipelineShardMetadata(
        model_meta=model_meta,
        device_rank=0,
        world_size=1,
        n_layers=model_meta.n_layers,
        start_layer=0,
        end_layer=model_meta.n_layers,
    )
    instance_id = InstanceId()

    print(f"{model_shard_meta=}")

    supervisor = await RunnerSupervisor.create(
        model_shard_meta=model_shard_meta,
        hosts=hosts(1, offset=10),
    )

    try:
        full_response = ""

        task = chat_completion_task(instance_id, TaskId())
        task.task_params.messages[0].content = lorem_ipsum * 3


        async for chunk in supervisor.stream_response(
            task=task
        ):
            if isinstance(chunk, TokenChunk):
                full_response += chunk.text

        assert len(full_response) > 100

    finally:
        await supervisor.astop()


@pytest.mark.asyncio
async def test_supervisor_two_node_long_prompt_response(
    pipeline_shard_meta: Callable[..., PipelineShardMetadata],
    hosts: Callable[..., list[Host]],
    chat_completion_task: Callable[[InstanceId, TaskId], Task],
    lorem_ipsum: str,
    logger: Logger,
):
    """Test two-node long prompt inference"""
    logger_test_install(logger)
    instance_id = InstanceId()

    async def create_supervisor(shard_idx: int) -> RunnerSupervisor:
        model_meta = MODEL_CARDS['llama-3.2-1b'].metadata
        model_shard_meta = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=shard_idx,
            world_size=2,
            n_layers=model_meta.n_layers,
            start_layer=0 if shard_idx == 0 else model_meta.n_layers // 2,
            end_layer=model_meta.n_layers // 2 if shard_idx == 0 else model_meta.n_layers,
        )
        supervisor = await RunnerSupervisor.create(
            model_shard_meta=model_shard_meta,
            hosts=hosts(2, offset=15),
        )
        return supervisor

    create_supervisor_0 = asyncio.create_task(create_supervisor(0))
    create_supervisor_1 = asyncio.create_task(create_supervisor(1))
    supervisor_0, supervisor_1 = await asyncio.gather(
        create_supervisor_0, create_supervisor_1
    )

    await asyncio.sleep(0.1)

    try:
        full_response_0 = ""
        full_response_1 = ""
        stop_reason_0: FinishReason | None = None
        stop_reason_1: FinishReason | None = None

        task = chat_completion_task(instance_id, TaskId())
        task.task_params.messages[0].content = lorem_ipsum * 3

        async def collect_response_0():
            nonlocal full_response_0, stop_reason_0
            async for chunk in supervisor_0.stream_response(task=task):
                if isinstance(chunk, TokenChunk):
                    full_response_0 += chunk.text
                    if chunk.finish_reason:
                        stop_reason_0 = chunk.finish_reason

        async def collect_response_1():
            nonlocal full_response_1, stop_reason_1
            async for chunk in supervisor_1.stream_response(task=task):
                if isinstance(chunk, TokenChunk):
                    full_response_1 += chunk.text
                    if chunk.finish_reason:
                        stop_reason_1 = chunk.finish_reason

        # Run both stream responses simultaneously
        _ = await asyncio.gather(collect_response_0(), collect_response_1())

        assert len(full_response_0) > 100
        assert len(full_response_1) > 100

    finally:
        await supervisor_0.astop()
        await supervisor_1.astop()

