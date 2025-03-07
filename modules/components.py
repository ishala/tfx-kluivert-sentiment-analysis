import os
import json
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)


def init_components(
    data_dir,
    transform_module,
    training_module,
    tuner_module,
    training_steps,
    eval_steps,
    serving_model_dir,
):

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=output
    )

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )

    tuner = Tuner(
        module_file=os.path.abspath(tuner_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
    )

    trainer = Trainer(
        module_file=training_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval']),
        hyperparameters=tuner.outputs['best_hyperparameters']
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='polarity')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),

                # AUC untuk multiclass dengan JSON encoding
                tfma.MetricConfig(class_name='AUC',
                                  config=json.dumps({'curve': 'ROC'})),

                # Confusion Matrix untuk melihat distribusi prediksi per kelas
                tfma.MetricConfig(class_name='MultiClassConfusionMatrixPlot'),

                # Akurasi Kategorikal (one-hot encoding)
                tfma.MetricConfig(class_name='CategoricalAccuracy',
                                  threshold=tfma.MetricThreshold(
                                      value_threshold=tfma.GenericValueThreshold(
                                          lower_bound={'value': 0.1})  # Lebih rendah
                                  )
                                  ),

                # Akurasi Sparse Categorical jika labelnya integer
                tfma.MetricConfig(class_name='SparseCategoricalAccuracy'),

                # Top-K Akurasi
                # tfma.MetricConfig(class_name='TopKCategoricalAccuracy', config=json.dumps({'k': 5})),

                # Precision dan Recall
                tfma.MetricConfig(class_name='Precision'),
                tfma.MetricConfig(class_name='Recall'),
            ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=os.path.join(
                    serving_model_dir,
                    'sentiment-analysis-model'))))

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )

    return components
