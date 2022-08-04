//===- MLIRToyVisitor.h ---------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file is the visitor for the MLIR Toy language AST.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOY_VISITOR_H
#define MLIR_TOY_VISITOR_H

#include "ToyBaseVisitor.h"
#include "ToyLexer.h"
#include "ToyParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "toy/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include <utility>

class MLIRToyVisitor : public ToyBaseVisitor {
public:
  /// AST Visitor Constructor
  MLIRToyVisitor(std::string filename, mlir::MLIRContext &context)
      : builder(&context), fileName(filename) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  /// Get the MLIR module.
  mlir::ModuleOp getModule() { return theModule; }

private:
  /// The Top Level MLIR Module Operation
  /// The module contains all the components generated from the source toy file.
  mlir::ModuleOp theModule;
  /// The MLIR Operations Builder
  /// The builder helps create MLIR operations when traversing the AST.
  mlir::OpBuilder builder;
  /// The Symbol Table
  llvm::ScopedHashTable<llvm::StringRef, std::pair<mlir::Value, ToyParser::VarDeclContext*>> symbolTable;
  llvm::ScopedHashTable<llvm::StringRef, int> funSymbolTable;
  // Register the filename for the string attribute in MLIR location object.
  std::string fileName;
  /// A mapping for named struct types to the underlying MLIR type and the
  /// original AST node.
  llvm::StringMap<mlir::Type> structTypeMap;
  llvm::StringMap<ToyParser::StructDefineContext*> structCtxMap;



  /// Declare a variable in the current scope
  /// - Check if the variable is already registered.
  /// -  Register variable in the symbol table.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value, ToyParser::VarDeclContext* ctx) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, {value, ctx});
    return mlir::success();
  }

  mlir::LogicalResult funcDeclare(llvm::StringRef functionName, int argsNumber) {
    if (funSymbolTable.count(functionName))
      return mlir::failure();
    funSymbolTable.insert(functionName, argsNumber);
    return mlir::success();
  }

  /// Location
  /// Get the MLIR location object with the current line and row of the toy
  /// source file.
  mlir::Location loc(int line, int row) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(fileName), line,
                                     row);
  }

  /// Get the tensor type according to the shape.
  mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  mlir::Type getType(std::string typeName, mlir::Location location) {
    auto it = structTypeMap.find(typeName);
    if (it == structTypeMap.end()) {
      mlir::emitError(location) << "error: unknown struct type '" << typeName << "'";
      return nullptr;
    }
    return it->second;
  }

  // Get the tensor value from the tensor literal node.
  std::any getTensor(ToyParser::TensorLiteralContext *ctx) {
    // [TODO][HIGH] find a better way to define the `dims`.
    std::vector<int64_t> dims;
    // get dimensions.
    dims.push_back(ctx->Comma().size() + 1);
    if (ctx->tensorLiteral(0)->tensorLiteral(0)) {
      dims.push_back(ctx->tensorLiteral(0)->Comma().size() + 1);
    }
    mlir::Type elementType = builder.getF64Type();
    auto type = getType(dims);
    auto dataType = mlir::RankedTensorType::get(dims, elementType);
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(ctx->data));
    auto loaction =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value value =
        builder.create<mlir::toy::ConstantOp>(loaction, type, dataAttribute);
    return value;
  }

  mlir::DenseElementsAttr getConstantAttr(ToyParser::TensorLiteralContext* ctx) {
    std::vector<int64_t> dims;
    // get dimensions.
    dims.push_back(ctx->Comma().size()+1);
    if (ctx->tensorLiteral(0)->tensorLiteral(0)) {
      auto list = ctx->tensorLiteral(0);
      while(list) {
        dims.push_back(list->Comma().size() + 1);
        if (list->tensorLiteral(0) && list->tensorLiteral(0)->Comma().size()) 
          list = list->tensorLiteral(0);
        else 
          break;
      }
    }
    mlir::Type elementType = builder.getF64Type();
    auto type = getType(dims);
    auto dataType = mlir::RankedTensorType::get(dims, elementType);
    return
    mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(ctx->data));
  }

  mlir::DenseElementsAttr getConstantAttr(antlr4::tree::TerminalNode* node) {
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get({}, elementType);
    double number = std::stod(node->toString());
    return mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(number));
  }




  std::pair<mlir::ArrayAttr, mlir::Type> 
  getConstantAttr(ToyParser::StructLiteralContext *ctx) {
    std::vector<mlir::Attribute> attrElements;
    std::vector<mlir::Type> typeElements;
    for (auto structContext : ctx->structLiteral()) {
      if (structContext->BracketOpen()) {
        auto attrTypePair = getConstantAttr(structContext);
        attrElements.push_back(attrTypePair.first);
        typeElements.push_back(attrTypePair.second);
      }
    }
    /*if (ctx->literalList()) {
      auto list = ctx->literalList();
      while (list) {
        if (list->tensorLiteral()) {
          if (list->tensorLiteral()->Number()) {
            attrElements.push_back(getConstantAttr(list->tensorLiteral()->Number()));
            typeElements.push_back(getType(llvm::None));
          } else {
            attrElements.push_back(getConstantAttr(list->tensorLiteral()));
            typeElements.push_back(getType(llvm::None));
          }
        }
        if (list->literalList())
          list = list->literalList();
        else 
          break;
      }
    }*/

    for (auto tensor : ctx->literalList()) {
      if (tensor->tensorLiteral()) {
        if (tensor->tensorLiteral()->Number()) {
          attrElements.push_back(getConstantAttr(tensor->tensorLiteral()->Number()));
          typeElements.push_back(getType(llvm::None));
        } else {
          attrElements.push_back(getConstantAttr(tensor->tensorLiteral()));
          typeElements.push_back(getType(llvm::None));
        }
      }
    }

    mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
    mlir::Type dataType = mlir::toy::StructType::get(typeElements);
    return std::make_pair(dataAttr, dataType);
  }

  virtual std::any visitModule(ToyParser::ModuleContext *ctx) override {
    llvm::ScopedHashTableScope<llvm::StringRef, int> protoTypeSymbolTable(funSymbolTable);
    for (auto& function : ctx->funDefine()) {
      auto protoType =  function->prototype();
      auto functionName =  protoType->Identifier()->toString();
      auto declNumber = 0;
      if (protoType->declList()) {
        auto list = protoType->declList();
        while (list) {
          declNumber++;
          if (list->declList())
            list = list->declList();
          else 
            break;
        }
      }
      funcDeclare(function->prototype()->idName, declNumber);
    } 
    return visitChildren(ctx);
  }

  /// Function Definition Visitor
  /// - Register the function name, argument list, and return value into the
  /// symbol table.
  /// - Visit function prototype.
  /// - Visit fucntion block.
  /// - Process the return operation.
  virtual std::any visitFunDefine(ToyParser::FunDefineContext *ctx) override {
    // [TODO] make the function support argument list and return value.
    llvm::ScopedHashTableScope<llvm::StringRef, std::pair<mlir::Value, ToyParser::VarDeclContext*>> varScope(
        symbolTable);
    builder.setInsertionPointToEnd(theModule.getBody());
    // Visit function prototype.
    mlir::toy::FuncOp function = std::any_cast<mlir::toy::FuncOp>(visit(ctx->prototype()));
    mlir::Block &entryBlock = function.front();
    builder.setInsertionPointToStart(&entryBlock);

    std::vector<std::string> args;
    std::vector<ToyParser::VarDeclContext*> varDecls;
   if (ctx->prototype()->declList()) {
      auto list = ctx->prototype()->declList();
       while (list->varDecl()) {
        args.push_back(list->varDecl()->idName);
        varDecls.push_back(list->varDecl());
        if (list->declList())
          list = list->declList();
        else
          break;
      }
    }

    llvm::ArrayRef<std::string> protoArgs = args;
    llvm::ArrayRef<ToyParser::VarDeclContext*> protoVarDecls = varDecls;
    for (auto value : llvm::zip(protoArgs, entryBlock.getArguments(), protoVarDecls)) {
      declare(std::get<0>(value), std::get<1>(value), std::get<2>(value));
    }

    // Visit fucntion block.
    visit(ctx->block());
    // Check the return status.
    // If there is no return expression at the end of the function, it will
    // generate a return operation automatically.
    mlir::toy::ReturnOp returnOp;
    if (!entryBlock.empty()) 
      returnOp = llvm::dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());
    if (!returnOp) {
      auto location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      builder.create<mlir::toy::ReturnOp>(location);
    } else if(returnOp.hasOperand()) {
      std::vector<int64_t> shape;
      function.setType(builder.getFunctionType(function.getFunctionType().getInputs(),getType(shape)));
    } 
    if (ctx->prototype()->Identifier()->toString() != "main") 
      function.setPrivate();
    return 0;
  }

  /// Prototype Visitor
  virtual std::any visitPrototype(ToyParser::PrototypeContext *ctx) override {
    mlir::Location location =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    auto varNumber = 0;
    // Get the number of arguments.
    if (ctx->declList()) {
      auto list = ctx->declList();
      while (list->varDecl()) {
        varNumber++;
        if (list->declList())
          list = list->declList();
        else
          break;
      }
    }
    llvm::SmallVector<mlir::Type, 4> argTypes;
    if (ctx->declList()) {
      auto list = ctx->declList();
      while (list->varDecl()) {
        if (list->varDecl()->Identifier().size() == 2) {
          auto location = loc(list->varDecl()->start->getLine(), list->varDecl()->start->getCharPositionInLine());
          auto type = getType(list->varDecl()->Identifier(0)->toString(),location);
          argTypes.push_back(type);
        } else {
          auto type = getType({});
          argTypes.push_back(type);
        }
        if (list->declList()) 
          list = list->declList();
        else 
          break;
      }
    } 
    auto funType = builder.getFunctionType(argTypes, llvm::None);
    auto func = builder.create<mlir::toy::FuncOp>(
        location, ctx->Identifier()->toString(), funType);
    return func;
  }

  virtual std::any visitStructLiteral(ToyParser::StructLiteralContext *ctx) override {
    mlir::ArrayAttr dataAttr;
    mlir::Type dataType;
    std::tie(dataAttr, dataType) = getConstantAttr(ctx);
    mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value value = builder.create<mlir::toy::StructConstantOp>(location, dataType, dataAttr);
    return value;
  }
  ToyParser::StructDefineContext* getStructFor(ToyParser::ExpressionContext* ctx) {    
    std::string structName;
    if (ctx->Dot()) {
     auto memberName = ctx->expression(1)->identifierExpr()->Identifier()->toString();
     auto parentCtx = getStructFor(ctx->expression(0));
     for (auto varDecl : parentCtx->varDecl()) {
      if (varDecl->idName == memberName) {
        structName = varDecl->Identifier(0)->toString();
        break;
      }
     }
    } else {
      auto varDecl = symbolTable.lookup(ctx->identifierExpr()->Identifier()->toString()).second;
      structName = varDecl->Identifier(0)->toString();
    } 
    return structCtxMap.lookup(structName);
  }

  llvm::Optional<size_t> getMemberIndex(ToyParser::ExpressionContext* ctx) {
    auto accessIndex = 0;
    auto parentCtx = getStructFor(ctx->expression(0));
    auto memberName = ctx->expression(1)->identifierExpr()->Identifier()->toString();
    for (auto vardecl : parentCtx->varDecl()) {
      if (vardecl->idName == memberName) 
        break;
      accessIndex++;
    }
    return accessIndex;
  }

  /// Expression Visitor
  /// - If the expression is tensor literal, return the tensor MLIR value.
  /// - If the expression is function call or variable, visit the identifier.
  /// - If the expression is add expression or mul expression return add or mul value.
  virtual std::any visitExpression(ToyParser::ExpressionContext *ctx) override {
    mlir::Value value;
    if (ctx->tensorLiteral()) {
      return getTensor(ctx->tensorLiteral());
    } else if (ctx->identifierExpr()) {
      return visit(ctx->identifierExpr());
    } else if (ctx->Add() || ctx->Mul()) {
      mlir::Value lhs = std::any_cast<mlir::Value>(visit(ctx->expression(0)));
      mlir::Value rhs = std::any_cast<mlir::Value>(visit(ctx->expression(1)));
      auto loaction = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      if (ctx->Add())
      value = builder.create<mlir::toy::AddOp>(loaction, lhs, rhs);  
      else 
      value = builder.create<mlir::toy::MulOp>(loaction, lhs, rhs);
      return value;
    } else if (ctx->Number()) {
      mlir::Location location = loc(ctx->Number()->getSymbol()->getLine(), ctx->Number()->getSymbol()->getCharPositionInLine());
      auto number = std::stod(ctx->Number()->toString());
      value = builder.create<mlir::toy::ConstantOp>(location, number);
      return value;
    } else if (ctx->structLiteral()) {
      return visit(ctx->structLiteral());
    } else if (ctx->Dot()) {
      value = std::any_cast<mlir::Value>(visit(ctx->expression(0)));
      mlir::Location location = loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      llvm::Optional<size_t> accessIndex = getMemberIndex(ctx);
      value = builder.create<mlir::toy::StructAccessOp>(location, value, *accessIndex);
      return value;
    }
    return value;
  }

  /// Variable Declaration Visitor
  /// - If the variable has the shape attribute, create the reshape operation.
  /// - Register the variable into the symbol table.
  virtual std::any visitVarDecl(ToyParser::VarDeclContext *ctx) override {
    // Get the variable MLIR value.
    mlir::Value value = std::any_cast<mlir::Value>(visit(ctx->expression()));
    // If the variable has the shape attribute, create the reshape operation.
    if (ctx->type()) {
      // [TODO][HIGH] try to use a better way to create the shape layout.
      std::vector<int64_t> v0;
      auto v1 = ctx->type()->Number();
      for (auto i : v1) {
        auto j = atoi(i->toString().c_str());
        v0.push_back(j);
      }
      mlir::Location location =
          loc(ctx->Identifier(0)->getSymbol()->getLine(),
              ctx->Identifier(0)->getSymbol()->getCharPositionInLine());
      value =
          builder.create<mlir::toy::ReshapeOp>(location, getType(v0), value);
    }
    /// Register the variable into the symbol table.
    mlir::failed(declare(ctx->idName, value, ctx));
    return 0;
  }

  /// Identifier Expression Visitor
  /// - Process function call.
  /// - Process variables.
  virtual std::any
  visitIdentifierExpr(ToyParser::IdentifierExprContext *ctx) override {
    mlir::Value value;
    auto argsNumber = 0;
    // If the identifier is a function call, visit and register all the
    // arguments. [TODO][LOW] add the semantic check (look up the symbol table)
    // for the function call.
    if (ctx->ParentheseOpen()) {
      auto location =
          loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
      llvm::SmallVector<mlir::Value, 4> oprands;
      for (auto i : ctx->expression()) {
        mlir::Value arg = std::any_cast<mlir::Value>(visit(i));
        oprands.push_back(arg);
        argsNumber++;
      }
      // If function call is a built-in operation, create the corresponding
      // operation.
      if (ctx->Identifier()->toString() == "print") {
        auto arg = oprands[0];
        builder.create<mlir::toy::PrintOp>(location, arg);
        return 0;
      } else if (ctx->Identifier()->toString() == "transpose") {
        auto arg = oprands[0];
        value = builder.create<mlir::toy::TransposeOp>(location, arg);
        return value;
      }
      auto numberdecl = funSymbolTable.lookup(ctx->Identifier()->toString());
      if (numberdecl != argsNumber) {
        if (funSymbolTable.count(ctx->Identifier()->toString()) == 0) 
          std::cout << "函数不存在" << std::endl;
        std::cout <<  ctx->Identifier()->toString() + "参数不匹配" << std::endl;
        exit(0);
      }     

      // If the function call cannot be mapped to the built-in operation, create
      // the GenericCallOp.
      value = builder.create<mlir::toy::GenericCallOp>(
          location, ctx->Identifier()->toString(), oprands);
      return value;
    } else {
      // If the identifier is a variable, return the MLIR value from the symbol
      // table.
      auto value = symbolTable.lookup(ctx->Identifier()->toString());
      return value.first;
    }
  }

  /// Return Expression Visitor
  virtual std::any visitReturnExpr(ToyParser::ReturnExprContext *ctx) override {
    auto location =
        loc(ctx->start->getLine(), ctx->start->getCharPositionInLine());
    mlir::Value expr = nullptr;
    if (ctx->expression()) {
      expr = std::any_cast<mlir::Value>(visit(ctx->expression()));
    }
    // Generate return operation based on whether the function has the return
    // value.
    builder.create<mlir::toy::ReturnOp>(location,
                                            expr ? llvm::makeArrayRef(expr)
                                         : llvm::ArrayRef<mlir::Value>());
    return 0;
  }

   virtual std::any visitStructDefine(ToyParser::StructDefineContext *ctx) override {
    std::vector<mlir::Type> elementTypes;
    for (auto variable : ctx->varDecl()) {
      mlir::Type type;
      if (variable->Identifier(1)) {   
        auto location = loc(variable->start->getLine(), variable->start->getCharPositionInLine());
        type = getType(variable->Identifier(0)->toString(), location);
      } else {
        type = getType({});   
      }
      elementTypes.push_back(type);
    }
    structTypeMap.try_emplace(ctx->Identifier()->toString(), mlir::toy::StructType::get(elementTypes));
    structCtxMap.try_emplace(ctx->Identifier()->toString(), ctx);
    return 0;
  }
};

#endif // MLIR_TOY_VISITOR_H
