/**
 * node-simple-fann - A Node.js wrapper for FANN (Fast Artificial Neural
 * Network Library) that uses MongoDB to store the models
 *
 * Copyright (C) 2014  Fredrik Söderström <tirithen@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

var mongodb = require('mongodb'),
    mkdirp = require('mkdirp'),
    fann = require('fann'),
    path = require('path'),
    fs = require('fs'),
    ObjectID = mongodb.ObjectID,
    handler = new NeuralNetworkHandler();

// TODO: If error has changed to little between trainings, abort and try with a more complex network
// TODO: If error sometimes increase, abort and try with a less complex network
// TODO: Possibility to vary the number of layers

function Model(options) {
    this._document = options.document;
    options.document = options.document || {};
    this._handler = options.handler;
    this._id = options._id || options.document._id;
    this.name = options.name || options.document.name;
    this.layers = options.layers || options.document.layers;
    this.epochs = options.epochs || options.document.epochs;
    this.error = options.error || options.document.error;

    if (!(this._id instanceof ObjectID)) {
        this._id = new ObjectID(this._id);
    }

    if (!Array.isArray(this.layers) || this.layers.length < 3) {
        throw new Error('Layers array attribute needs to be at least 3');
    }

    this.filename = __dirname + '/neuralNetwork/' + this.name + '.nnet';
    if (fs.existsSync(this.filename)) {
        this.neuralNetwork = new fann.load(this.filename);
    }
}

Model.prototype.save = function (callback) {
    var self = this;

    this._handler.databaseConnection
        .collection('models')
        .update(
            { _id: this._id },
            {
                _id: this._id,
                name: this.name,
                layers: this.layers,
                epochs: this.epochs,
                error: this.error
            },
            { upsert: true },
            function (error) {
                if (error) {
                    if (callback instanceof Function) {
                        callback(error);
                    }
                } else {
                    mkdirp(path.dirname(self.filename), function (error) {
                        if (error) {
                            if (callback instanceof Function) {
                                callback(error);
                            }
                        } else if (self.neuralNetwork) {
                            console.log('Saving network to ' + self.filename);

                            self.neuralNetwork.save(self.filename);
                            if (callback instanceof Function) {
                                callback();
                            }
                        }
                    });
                }
            }
        );
};

Model.prototype.train = function (callback) {
    var self = this;

    console.log('Training ' + this.name + '...');
    this._handler.databaseConnection
        .collection('traningData-' + this.name)
        .find()
        .toArray(function (error, documents) {
            var data = documents.map(function (document) {
                    return [ document.input, document.output ];
                }),
                neuralNetwork = new fann.shortcut(
                    self.layers[0],
                    self.layers[1],
                    self.layers[2]
                );

            neuralNetwork.train(data, {
                error: self.error,
                epochs: self.epochs
            });

            // TODO: only save the new neural network if it was better than the last one
            self.neuralNetwork = neuralNetwork;
            self.save(callback);

            if (callback instanceof Function) {
                callback(error);
            }
        });
};

Model.prototype.run = function (input) {
    if (!this.neuralNetwork) {
        throw new Error('No neural network to run through, make sure to train first');
    }

    return this.neuralNetwork.run(input);
};

Model.prototype.addTrainingData = function (input, output, rawData, callback) {
    this._handler.databaseConnection
        .collection('traningData-' + this.name)
        .insert(
            {
                input: input,
                output: output,
                rawData: rawData
            },
            function (error) {
                if (callback instanceof Function) {
                    callback(error);
                }
            }
        );
};

function NeuralNetworkHandler(connectionString) {
    this.databaseConnection = null;
    this.models = {};
}

NeuralNetworkHandler.prototype.loadModels = function (callback) {
    var self = this;

    this.databaseConnection.collection('models').find().toArray(function (error, documents) {
        documents.forEach(function (document) {
            self.models[document.name] = new Model({
            document: document,
            handler: self
            });
        });

        if (callback instanceof Function) {
            callback(error, self.models);
        }
    });
};

NeuralNetworkHandler.prototype.addModel = function (options, callback) {
    var model;

    options.handler = this;
    model = new Model(options);
    model.save(callback);
    this.models[model.name] = model;
};

NeuralNetworkHandler.prototype.databaseConnect = function (connectionString, callback) {
    var self = this;

    mongodb.connect(connectionString, function (error, databaseConnection) {
        if (error) {
            throw error;
        }

        self.databaseConnection = databaseConnection;
        self.loadModels(callback);
    });
};

module.exports = function (connectionString, callback) {
    handler.databaseConnect(connectionString, callback);

    return handler;
};
