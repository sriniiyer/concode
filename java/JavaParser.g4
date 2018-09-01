compilationUnit
    : packageDeclaration ?  importDeclaration *  typeDeclaration *  EOF
     ;

identifierNT: IDENTIFIER ;

packageDeclaration
    : annotation *  PACKAGE qualifiedName ';'
     ;

importDeclaration
    : IMPORT STATIC ?  qualifiedName  ( '.' '*' )  ?  ';'
     ;

typeDeclaration
    : classOrInterfaceModifier * 
       ( classDeclaration | enumDeclaration | interfaceDeclaration | annotationTypeDeclaration ) 
    | ';'
     ;

modifier
    : classOrInterfaceModifier
    | NATIVE
    | SYNCHRONIZED
    | TRANSIENT
    | VOLATILE
     ;

classOrInterfaceModifier
    : annotation
    | PUBLIC
    | PROTECTED
    | PRIVATE
    | STATIC
    | ABSTRACT
    | FINAL  
    | STRICTFP
     ;

variableModifier
    : FINAL
    | annotation
     ;

classDeclaration
    : CLASS identifierNT typeParameters ? 
       ( EXTENDS typeType )  ? 
       ( IMPLEMENTS typeList )  ? 
      classBody
     ;

typeParameters
    : '<' typeParameter  ( ',' typeParameter )  *  '>'
     ;

typeParameter
    : annotation *  identifierNT  ( EXTENDS typeBound )  ? 
     ;

typeBound
    : typeType  ( '&' typeType )  * 
     ;

enumDeclaration
    : ENUM identifierNT  ( IMPLEMENTS typeList )  ?  '{' enumConstants ?  ',' ?  enumBodyDeclarations ?  '}'
     ;

enumConstants
    : enumConstant  ( ',' enumConstant )  * 
     ;

enumConstant
    : annotation *  identifierNT arguments ?  classBody ? 
     ;

enumBodyDeclarations
    : ';' classBodyDeclaration * 
     ;

interfaceDeclaration
    : INTERFACE identifierNT typeParameters ?   ( EXTENDS typeList )  ?  interfaceBody
     ;

classBody
    : '{' classBodyDeclaration *  '}'
     ;

interfaceBody
    : '{' interfaceBodyDeclaration *  '}'
     ;

classBodyDeclaration
    : ';'
    | STATIC ?  block
    | modifier *  memberDeclaration
     ;

memberDeclaration
    : methodDeclaration
    | genericMethodDeclaration
    | fieldDeclaration
    | constructorDeclaration
    | genericConstructorDeclaration
    | interfaceDeclaration
    | annotationTypeDeclaration
    | classDeclaration
    | enumDeclaration
     ;

methodDeclaration
    : typeTypeOrVoid identifierNT formalParameters  ( '[' ']' )  * 
       ( THROWS qualifiedNameList )  ? 
      methodBody
     ;

methodBody
    : block
    | ';'
     ;

typeTypeOrVoid
    : typeType
    | VOID
     ;

genericMethodDeclaration
    : typeParameters methodDeclaration
     ;

genericConstructorDeclaration
    : typeParameters constructorDeclaration
     ;

constructorDeclaration
    : identifierNT formalParameters  ( THROWS qualifiedNameList )  ?  constructorBody=block
     ;

fieldDeclaration
    : typeType variableDeclarators ';'
     ;

interfaceBodyDeclaration
    : modifier *  interfaceMemberDeclaration
    | ';'
     ;

interfaceMemberDeclaration
    : constDeclaration
    | interfaceMethodDeclaration
    | genericInterfaceMethodDeclaration
    | interfaceDeclaration
    | annotationTypeDeclaration
    | classDeclaration
    | enumDeclaration
     ;

constDeclaration
    : typeType constantDeclarator  ( ',' constantDeclarator )  *  ';'
     ;

constantDeclarator
    : identifierNT  ( '[' ']' )  *  '=' variableInitializer
     ;

interfaceMethodDeclaration
    : interfaceMethodModifier *   ( typeTypeOrVoid | typeParameters annotation *  typeTypeOrVoid ) 
      identifierNT formalParameters  ( '[' ']' )  *   ( THROWS qualifiedNameList )  ?  methodBody
     ;

interfaceMethodModifier
    : annotation
    | PUBLIC
    | ABSTRACT
    | DEFAULT
    | STATIC
    | STRICTFP
     ;

genericInterfaceMethodDeclaration
    : typeParameters interfaceMethodDeclaration
     ;

variableDeclarators
    : variableDeclarator  ( ',' variableDeclarator )  * 
     ;

variableDeclarator
    : variableDeclaratorId  ( '=' variableInitializer )  ? 
     ;

variableDeclaratorId
    : identifierNT  ( '[' ']' )  * 
     ;

variableInitializer
    : arrayInitializer
    | expression
     ;

arrayInitializer
    : '{'  ( variableInitializer  ( ',' variableInitializer )  *   ( ',' )  ?   )  ?  '}'
     ;

classOrInterfaceType
    : identifierNT typeArguments ?   ( '.' identifierNT typeArguments ?  )  * 
     ;

typeArgument
    : typeType
    | '?'  (  ( EXTENDS | SUPER )  typeType )  ? 
     ;

qualifiedNameList
    : qualifiedName  ( ',' qualifiedName )  * 
     ;

formalParameters
    : '(' formalParameterList ?  ')'
     ;

formalParameterList
    : formalParameter  ( ',' formalParameter )  *   ( ',' lastFormalParameter )  ? 
    | lastFormalParameter
     ;

formalParameter
    : variableModifier *  typeType variableDeclaratorId
     ;

lastFormalParameter
    : variableModifier *  typeType '...' variableDeclaratorId
     ;

qualifiedName
    : identifierNT  ( '.' identifierNT )  * 
     ;

literal
    : integerLiteral
    | floatLiteral
    | nt_char_literal
    | nt_string_literal
    | nt_bool_literal
    | nt_null_literal
     ;

nt_char_literal: CHAR_LITERAL;
nt_string_literal: STRING_LITERAL;
nt_bool_literal: BOOL_LITERAL;
nt_null_literal: NULL_LITERAL;
nt_decimal_literal: DECIMAL_LITERAL;
nt_hex_literal: HEX_LITERAL;
nt_oct_literal: OCT_LITERAL;
nt_binary_literal: BINARY_LITERAL;
nt_float_literal: FLOAT_LITERAL;
nt_hex_float_literal: HEX_FLOAT_LITERAL;

integerLiteral
    : nt_decimal_literal
    | nt_hex_literal
    | nt_oct_literal
    | nt_binary_literal
     ;

floatLiteral
    : nt_float_literal
    | nt_hex_float_literal
     ;

annotation
    : '@' qualifiedName  ( '('  (  elementValuePairs | elementValue  )  ?  ')' )  ? 
     ;

elementValuePairs
    : elementValuePair  ( ',' elementValuePair )  * 
     ;

elementValuePair
    : identifierNT '=' elementValue
     ;

elementValue
    : expression
    | annotation
    | elementValueArrayInitializer
     ;

elementValueArrayInitializer
    : '{'  ( elementValue  ( ',' elementValue )  *  )  ?   ( ',' )  ?  '}'
     ;

annotationTypeDeclaration
    : '@' INTERFACE identifierNT annotationTypeBody
     ;

annotationTypeBody
    : '{'  ( annotationTypeElementDeclaration )  *  '}'
     ;

annotationTypeElementDeclaration
    : modifier *  annotationTypeElementRest
    | ';' 
     ;

annotationTypeElementRest
    : typeType annotationMethodOrConstantRest ';'
    | classDeclaration ';' ? 
    | interfaceDeclaration ';' ? 
    | enumDeclaration ';' ? 
    | annotationTypeDeclaration ';' ? 
     ;

annotationMethodOrConstantRest
    : annotationMethodRest
    | annotationConstantRest
     ;

annotationMethodRest
    : identifierNT '(' ')' defaultValue ? 
     ;

annotationConstantRest
    : variableDeclarators
     ;

defaultValue
    : DEFAULT elementValue
     ;

block
    : '{' blockStatement *  '}'
     ;

blockStatement
    : localVariableDeclaration ';'
    | statement
    | localTypeDeclaration
     ;

localVariableDeclaration
    : variableModifier *  typeType variableDeclarators
     ;

localTypeDeclaration
    : classOrInterfaceModifier * 
       ( classDeclaration | interfaceDeclaration ) 
    | ';'
     ;

statement
    : blockLabel=block
    | ASSERT expression  ( ':' expression )  ?  ';'
    | IF parExpression statement  ( ELSE statement )  ? 
    | FOR '(' forControl ')' statement
    | WHILE parExpression statement
    | DO statement WHILE parExpression ';'
    | TRY block  ( catchClause + finallyBlock ?  | finallyBlock ) 
    | TRY resourceSpecification block catchClause *  finallyBlock ? 
    | SWITCH parExpression '{' switchBlockStatementGroup *  switchLabel *  '}'
    | SYNCHRONIZED parExpression block
    | RETURN expression ?  ';'
    | THROW expression ';'
    | BREAK identifierNT ?  ';'
    | CONTINUE identifierNT ?  ';'
    | SEMI
    | statementExpression=expression ';'
    | identifierLabel=identifierNT ':' statement
     ;

catchClause
    : CATCH '(' variableModifier *  catchType identifierNT ')' block
     ;

catchType
    : qualifiedName  ( '|' qualifiedName )  * 
     ;

finallyBlock
    : FINALLY block
     ;

resourceSpecification
    : '(' resources ';' ?  ')'
     ;

resources
    : resource  ( ';' resource )  * 
     ;

resource
    : variableModifier *  classOrInterfaceType variableDeclaratorId '=' expression
     ;

switchBlockStatementGroup
    : switchLabel + blockStatement +
     ;

switchLabel
    : CASE  ( constantExpression=expression | enumConstantName=identifierNT )  ':'
    | DEFAULT ':'
     ;

forControl
    : enhancedForControl
    | forInit ?  ';' expression ?  ';' forUpdate=expressionList ? 
     ;

forInit
    : localVariableDeclaration
    | expressionList
     ;

enhancedForControl
    : variableModifier *  typeType variableDeclaratorId ':' expression
     ;

parExpression
    : '(' expression ')'
     ;

expressionList
    : expression  ( ',' expression )  * 
     ;

expression
    : primary
    | expression '.'
       ( identifierNT
      | THIS
      | NEW nonWildcardTypeArguments ?  innerCreator
      | SUPER superSuffix
      | explicitGenericInvocation
       ) 
    | expression '[' expression ']'
    | expression '(' expressionList ?  ')'
    | NEW creator
    | '(' typeType ')' expression
    | expression ( '++' | '--' ) 
    |  ( '+'|'-'|'++'|'--' )  expression
    |  ( '~'|'!' )  expression
    | expression  ( '*'|'/'|'%' )  expression
    | expression  ( '+'|'-' )  expression
    | expression  ( '<' '<' | '>' '>' '>' | '>' '>' )  expression
    | expression  ( '<=' | '>=' | '>' | '<' )  expression
    | expression INSTANCEOF typeType
    | expression  ( '==' | '!=' )  expression
    | expression '&' expression
    | expression '^' expression
    | expression '|' expression
    | expression '&&' expression
    | expression '||' expression
    | expression '?' expression ':' expression
    | <assoc=right> expression
       ( '=' | '+=' | '-=' | '*=' | '/=' | '&=' | '|=' | '^=' | '>>=' | '>>>=' | '<<=' | '%=' ) 
      expression
    | lambdaExpression 

    | expression '::' typeArguments ?  identifierNT
    | typeType '::'  ( typeArguments ?  identifierNT | NEW ) 
    | classType '::' typeArguments ?  NEW
     ;

lambdaExpression
    : lambdaParameters '->' lambdaBody
     ;

lambdaParameters
    : identifierNT
    | '(' formalParameterList ?  ')'
    | '(' identifierNT  ( ',' identifierNT )  *  ')'
     ;

lambdaBody
    : expression
    | block
     ;

primary
    : '(' expression ')'
    | THIS
    | SUPER
    | literal
    | identifierNT
    | typeTypeOrVoid '.' CLASS
    | nonWildcardTypeArguments  ( explicitGenericInvocationSuffix | THIS arguments ) 
     ;

classType
    :  ( classOrInterfaceType '.' )  ?  annotation *  identifierNT typeArguments ? 
     ;

creator
    : nonWildcardTypeArguments createdName classCreatorRest
    | createdName  ( arrayCreatorRest | classCreatorRest ) 
     ;

createdName
    : identifierNT typeArgumentsOrDiamond ?   ( '.' identifierNT typeArgumentsOrDiamond ?  )  * 
    | primitiveType
     ;

innerCreator
    : identifierNT nonWildcardTypeArgumentsOrDiamond ?  classCreatorRest
     ;

arrayCreatorRest
    : '['  ( ']'  ( '[' ']' )  *  arrayInitializer | expression ']'  ( '[' expression ']' )  *   ( '[' ']' )  *  ) 
     ;

classCreatorRest
    : arguments classBody ? 
     ;

explicitGenericInvocation
    : nonWildcardTypeArguments explicitGenericInvocationSuffix
     ;

typeArgumentsOrDiamond
    : '<' '>'
    | typeArguments
     ;

nonWildcardTypeArgumentsOrDiamond
    : '<' '>'
    | nonWildcardTypeArguments
     ;

nonWildcardTypeArguments
    : '<' typeList '>'
     ;

typeList
    : typeType  ( ',' typeType )  * 
     ;

typeType
    : annotation ?   ( classOrInterfaceType | primitiveType )   ( '[' ']' )  * 
     ;

primitiveType
    : BOOLEAN
    | CHAR
    | BYTE
    | SHORT
    | INT
    | LONG
    | FLOAT
    | DOUBLE
     ;

typeArguments
    : '<' typeArgument  ( ',' typeArgument )  *  '>'
     ;

superSuffix
    : arguments
    | '.' identifierNT arguments ? 
     ;

explicitGenericInvocationSuffix
    : SUPER superSuffix
    | identifierNT arguments
     ;

arguments
    : '(' expressionList ?  ')'
     ;
